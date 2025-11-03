"""
SR Quality Model - LightGBM

Pure ML-based SR level quality prediction.
Replaces hand-crafted weighted scoring with data-driven predictions.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr

# Import HPO utilities
try:
    from src.utils.ml_common.optimization.hpo_utils import optimize_hyperparameters
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False

logger = logging.getLogger(__name__)


class SRQualityModel:
    """LightGBM model for predicting SR level quality.
    
    PURE ML APPROACH:
    - Trained on historical SR performance data
    - Predicts quality_score (0-1) from features
    - Replaces weighted composite scoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize SR quality model.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _get_default_config(self) -> Dict:
        """Default LightGBM configuration optimized for SR quality prediction.
        
        ANTI-OVERFITTING CONFIGURATION:
        - Strong regularization (L1=1.0, L2=1.0) to prevent overfitting
        - Reduced complexity (31 leaves, depth 5) for balanced model
        - Increased min_data_in_leaf (50) to require more evidence
        - min_gain_to_split (0.3) to prevent weak splits
        - Lower learning rate (0.03) for smoother convergence
        - Feature/bagging subsampling (0.7/0.7) to reduce variance
        """
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            
            # Balanced complexity (aligned with HPO defaults)
            'num_leaves': 31,  # Moderate tree structure
            'max_depth': 5,    # Moderate depth
            
            # Strong regularization
            'lambda_l1': 1.0,  # L1 regularization
            'lambda_l2': 1.0,  # L2 regularization
            
            # Require more data per leaf
            'min_data_in_leaf': 50,  # More conservative splits
            
            # Prevent weak splits that overfit noise
            'min_gain_to_split': 0.3,  # Minimum gain required to split
            
            # Lower learning rate for stability (was: 0.05)
            'learning_rate': 0.03,  # Slower, more stable learning
            
            # Increased subsampling to reduce overfitting
            'feature_fraction': 0.7,  # Use 70% of features per tree
            'bagging_fraction': 0.7,  # Use 70% of data per iteration
            'bagging_freq': 5,
            
            'verbose': -1,
            'seed': 42,
            'force_col_wise': True
        }
    
    def train(self, training_data: pd.DataFrame, 
             target_column: str = 'quality_score',
             n_folds: int = 5,
             num_boost_round: int = 1000,
             early_stopping_rounds: int = 50) -> Dict:
        """Train LightGBM model with time series cross-validation.
        
        Args:
            training_data: DataFrame with features + quality_score
            target_column: Target column to predict
            n_folds: Number of CV folds
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Dictionary with CV scores and metrics
        """
        
        self.logger.info(f"🤖 Training SR Quality Model")
        self.logger.info(f"   Training samples (raw): {len(training_data)}")
        
        # FILTER OUT WEAK SR LEVELS (only train on meaningful levels)
        # Remove levels with very low quality or strength to improve signal/noise
        min_quality_threshold = 0.25  # Remove bottom 25% of levels
        
        # Filter by quality_score if available
        if 'quality_score' in training_data.columns:
            original_len = len(training_data)
            training_data = training_data[training_data['quality_score'] >= min_quality_threshold].copy()
            filtered_count = original_len - len(training_data)
            self.logger.info(f"   📊 Filtered out {filtered_count} weak levels (quality < {min_quality_threshold})")
        
        # Also filter by strength if available
        if 'feature_strength' in training_data.columns:
            original_len = len(training_data)
            training_data = training_data[training_data['feature_strength'] >= 0.4].copy()
            filtered_count = original_len - len(training_data)
            self.logger.info(f"   📊 Filtered out {filtered_count} low-strength levels (strength < 0.4)")
        
        self.logger.info(f"   ✅ Training samples (filtered): {len(training_data)}")
        
        if len(training_data) < 100:
            raise ValueError(f"Insufficient training data after filtering: {len(training_data)} samples")
        
        # Separate features from targets/metadata
        # If feature_names already set by train_with_hpo(), use those
        if self.feature_names is not None:
            self.logger.info(f"   ℹ️  Using pre-selected features from HPO: {len(self.feature_names)}")
            feature_cols = self.feature_names
        else:
            # CRITICAL FIX: Exclude ALL target/performance columns to prevent target leakage
            # Only use columns starting with 'feature_' as predictive features
            exclude_cols = ['date', 'symbol', 'exchange', 'timeframe', 'sample_weight',
                           # Primary target
                           'quality_score',
                           # Sub-targets (components of quality_score) - MUST EXCLUDE!
                           'bounce_quality', 'hold_quality', 'trade_quality', 
                           'speed_quality', 'volume_confirmation_quality',
                           # Performance metrics (forward-looking targets) - MUST EXCLUDE!
                           'hit_rate', 'bounce_strength', 'max_bounce_strength',
                           'hold_strength', 'trade_profit', 'rejection_speed', 'volume_quality']
            
            # SAFER: Use only columns that explicitly start with 'feature_'
            feature_cols = [c for c in training_data.columns 
                           if c.startswith('feature_') and not pd.isna(training_data[c]).all()]
            
            self.logger.info(f"\n   🔒 TARGET LEAKAGE PREVENTION:")
            self.logger.info(f"      Excluded {len(exclude_cols)} target/metadata columns")
            self.logger.info(f"      Using only 'feature_*' columns: {len(feature_cols)} features")
            
            if len(feature_cols) == 0:
                raise ValueError("No valid feature columns found! All columns start with 'feature_'")
            
            self.feature_names = feature_cols
        
        X = training_data[feature_cols]
        y = training_data[target_column]
        
        self.logger.info(f"   Features: {len(feature_cols)}")
        self.logger.info(f"   Target: {target_column}")
        self.logger.info(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")
        self.logger.info(f"   Target mean: {y.mean():.3f} ± {y.std():.3f}")
        
        # Handle missing values
        X = X.fillna(0.0)
        
        # Remove zero-variance features (provide no information)
        feature_variances = X.var()
        zero_var_features = feature_variances[feature_variances < 1e-10].index.tolist()
        if zero_var_features:
            self.logger.info(f"\n   🔧 Removing {len(zero_var_features)} zero-variance features:")
            for feat in zero_var_features[:5]:  # Show first 5
                self.logger.info(f"      - {feat}")
            if len(zero_var_features) > 5:
                self.logger.info(f"      ... and {len(zero_var_features) - 5} more")
            X = X.drop(columns=zero_var_features)
            self.feature_names = X.columns.tolist()
            self.logger.info(f"   ✅ Features after variance filter: {len(self.feature_names)}")
        
        # Get sample weights if available
        sample_weights = None
        if 'sample_weight' in training_data.columns:
            sample_weights = training_data['sample_weight'].values
            self.logger.info(f"\n   ✅ Using sample weights (soft filtering):")
            self.logger.info(f"      Weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
            self.logger.info(f"      Weight mean: {sample_weights.mean():.2f}")
        
        # STRATIFIED Time Series Cross-Validation
        # Bin quality scores into tiers to ensure balanced distribution across folds
        y_binned = pd.qcut(y, q=min(5, len(y)//10), labels=False, duplicates='drop')
        
        # Use TimeSeriesSplit but with stratification awareness
        # We'll manually adjust folds to ensure quality distribution
        tscv = TimeSeriesSplit(n_splits=n_folds)
        self.logger.info(f"\n   📊 Using Stratified Time-Series CV:")
        self.logger.info(f"      Quality bins: {y_binned.nunique()}")
        for bin_idx in range(y_binned.nunique()):
            bin_count = (y_binned == bin_idx).sum()
            bin_mean = y[y_binned == bin_idx].mean()
            self.logger.info(f"      Bin {bin_idx}: {bin_count} samples (avg quality: {bin_mean:.3f})")
        
        cv_scores = []
        fold_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"\n  📈 Training Fold {fold_idx + 1}/{n_folds}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Get weights for this fold
            if sample_weights is not None:
                weights_train = sample_weights[train_idx]
                weights_val = sample_weights[val_idx]
            else:
                weights_train = None
                weights_val = None
            
            self.logger.info(f"     Train: {len(X_train)} samples, Val: {len(X_val)} samples")
            
            # Create LightGBM datasets with sample weights
            train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
            val_data = lgb.Dataset(X_val, label=y_val, weight=weights_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.config,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
            )
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            fold_scores = {
                'fold': fold_idx,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'train_r2': r2_score(y_train, y_pred_train),
                'val_r2': r2_score(y_val, y_pred_val),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'val_mae': mean_absolute_error(y_val, y_pred_val),
                'num_boost_rounds': model.current_iteration()
            }
            
            cv_scores.append(fold_scores)
            fold_models.append(model)
            
            self.logger.info(f"     ✓ Val RMSE: {fold_scores['val_rmse']:.4f} | R²: {fold_scores['val_r2']:.4f} | MAE: {fold_scores['val_mae']:.4f}")
        
        # Select best model (lowest validation RMSE)
        best_idx = np.argmin([s['val_rmse'] for s in cv_scores])
        self.model = fold_models[best_idx]
        
        self.logger.info(f"\n🎯 Best Model: Fold {best_idx + 1}")
        self.logger.info(f"   Val RMSE: {cv_scores[best_idx]['val_rmse']:.4f}")
        self.logger.info(f"   Val R²: {cv_scores[best_idx]['val_r2']:.4f}")
        self.logger.info(f"   Val MAE: {cv_scores[best_idx]['val_mae']:.4f}")
        
        # Average CV scores
        avg_metrics = {
            'avg_val_rmse': np.mean([s['val_rmse'] for s in cv_scores]),
            'avg_val_r2': np.mean([s['val_r2'] for s in cv_scores]),
            'avg_val_mae': np.mean([s['val_mae'] for s in cv_scores]),
            'std_val_rmse': np.std([s['val_rmse'] for s in cv_scores]),
            'std_val_r2': np.std([s['val_r2'] for s in cv_scores])
        }
        
        self.logger.info(f"\n📊 Cross-Validation Summary:")
        self.logger.info(f"   Avg Val RMSE: {avg_metrics['avg_val_rmse']:.4f} ± {avg_metrics['std_val_rmse']:.4f}")
        self.logger.info(f"   Avg Val R²: {avg_metrics['avg_val_r2']:.4f} ± {avg_metrics['std_val_r2']:.4f}")
        self.logger.info(f"   Avg Val MAE: {avg_metrics['avg_val_mae']:.4f}")
        
        # Feature importance
        feature_importance_df = self._log_feature_importance()
        
        # Store metrics
        self.training_metrics = {
            'cv_scores': cv_scores,
            'best_fold': best_idx,
            'avg_metrics': avg_metrics,
            'config': self.config,
            'feature_importance': feature_importance_df.to_dict('records') if feature_importance_df is not None else []
        }
        
        # ========================================================================
        # COMPREHENSIVE MODEL QUALITY ASSESSMENT
        # ========================================================================
        self.logger.info(f"\n{'='*80}")
        self.logger.info("🔬 RUNNING COMPREHENSIVE MODEL QUALITY ASSESSMENT")
        self.logger.info(f"{'='*80}")
        
        try:
            from .model_quality_assessor import ModelQualityAssessor, FeatureImportanceAnalyzer
            from .comprehensive_reporter import ComprehensiveReporter
            
            # Get train/val metrics for final fold (best model)
            train_metrics_final = {
                'rmse': cv_scores[best_idx]['train_rmse'],
                'r2': cv_scores[best_idx]['train_r2'],
                'mae': cv_scores[best_idx]['train_mae']
            }
            val_metrics_final = {
                'rmse': cv_scores[best_idx]['val_rmse'],
                'r2': cv_scores[best_idx]['val_r2'],
                'mae': cv_scores[best_idx]['val_mae']
            }
            
            # Run quality assessment
            assessor = ModelQualityAssessor()
            
            # Get predictions for full dataset
            y_pred = self.model.predict(X)
            
            quality_assessment = {
                'timestamp': datetime.now().isoformat()
            }
            
            # 1. Overfitting detection
            quality_assessment['overfitting'] = assessor.detect_overfitting(
                train_metrics_final, val_metrics_final, cv_scores
            )
            
            # 2. Calibration
            quality_assessment['calibration'] = assessor.assess_calibration(y_pred, y.values)
            
            # 3. Prediction distribution
            quality_assessment['prediction_distribution'] = assessor.analyze_prediction_distribution(
                y_pred, y.values
            )
            
            # 4. Error analysis by bin
            quality_assessment['error_by_bin'] = assessor.analyze_errors_by_quality_bin(
                y_pred, y.values
            )
            
            # 5. Feature importance stability
            quality_assessment['feature_stability'] = assessor.analyze_feature_importance_stability(
                fold_models, self.feature_names
            )
            
            # Calculate overall health score
            health_score = assessor._calculate_health_score(quality_assessment)
            quality_assessment['health_score'] = health_score
            quality_assessment['production_ready'] = health_score >= 0.70
            
            # Feature importance analysis (multiple methods)
            importance_analyzer = FeatureImportanceAnalyzer()
            importance_analysis = importance_analyzer.calculate_all_importances(
                self.model, X, y, self.feature_names
            )
            
            # Generate comprehensive report
            reporter = ComprehensiveReporter(output_dir='outcomes')
            
            # Get symbol and timeframe from data
            symbol_val = training_data['symbol'].iloc[0] if 'symbol' in training_data.columns else 'UNKNOWN'
            timeframe_val = training_data['timeframe'].iloc[0] if 'timeframe' in training_data.columns else target_column
            
            report_paths = reporter.generate_complete_report(
                training_data=training_data,
                model=self.model,
                training_metrics=self.training_metrics,
                quality_assessment=quality_assessment,
                importance_analysis=importance_analysis,
                timeframe=timeframe_val,
                symbol=symbol_val
            )
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"✅ COMPREHENSIVE REPORTS GENERATED")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"\n📁 Output files:")
            self.logger.info(f"   📄 Report: {report_paths['markdown']}")
            self.logger.info(f"   📊 Levels CSV: {report_paths['csv']}")
            self.logger.info(f"   📋 Metrics JSON: {report_paths['json']}")
            
            # Store in training metrics
            self.training_metrics['quality_assessment'] = quality_assessment
            self.training_metrics['importance_analysis'] = {
                'shap_available': importance_analysis['shap_available'],
                'top_10_features': importance_analysis['combined_ranking'].head(10)['feature'].tolist()
            }
            self.training_metrics['report_paths'] = report_paths
            
        except ImportError as e:
            self.logger.warning(f"⚠️  Quality assessment modules not available: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error in quality assessment: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return self.training_metrics
    
    def train_with_hpo(self, training_data: pd.DataFrame,
                      target_column: str = 'quality_score',
                      filter_percentile: float = 80.0,
                      n_trials: int = 100,
                      n_folds: int = 5,
                      num_boost_round: int = 1000,
                      early_stopping_rounds: int = 50,
                      method: str = 'bayesian') -> Dict:
        """Train LightGBM model with Hyperparameter Optimization (HPO).
        
        Uses Bayesian optimization to find the best anti-overfitting configuration.
        NOW WITH DATA FILTERING: Trains only on top N% of quality levels!
        
        Args:
            training_data: DataFrame with features + quality_score
            target_column: Target column to predict
            filter_percentile: Keep top N% by quality (80 = top 20%)
            n_trials: Number of HPO trials (default: 100)
            n_folds: Number of CV folds
            num_boost_round: Max boosting rounds
            early_stopping_rounds: Early stopping patience
            method: Optimization method ('bayesian', 'staged', or 'multi_objective')
            
        Returns:
            Dictionary with best params, CV scores, and HPO results
        """
        if not HPO_AVAILABLE:
            self.logger.warning("⚠️ HPO not available, falling back to standard training")
            return self.train(training_data, target_column, n_folds, num_boost_round, early_stopping_rounds)
        
        self.logger.info(f"🎯 Training SR Quality Model with HPO ({n_trials} trials)")
        self.logger.info(f"   Training samples (raw): {len(training_data)}")
        
        # NEW APPROACH: NO HARD FILTERING - Use ONLY confidence weighting
        # Reason: Hard filtering removes variance → model collapses to predicting mean
        # Better: Keep ALL data but weight by quality (already in sample_weight column)
        
        # Optional: Light filtering for extreme outliers only (bottom 10%)
        if filter_percentile < 100.0 and filter_percentile >= 90.0:
            threshold = np.percentile(training_data[target_column], filter_percentile)
            original_len = len(training_data)
            training_data = training_data[training_data[target_column] >= threshold].copy()
            
            self.logger.info(f"\n📊 LIGHT FILTERING (remove bottom {filter_percentile:.0f}% outliers):")
            self.logger.info(f"   Quality threshold: {threshold:.3f}")
            self.logger.info(f"   Kept samples: {len(training_data):,} ({len(training_data)/original_len*100:.1f}%)")
            self.logger.info(f"   Removed outliers: {original_len - len(training_data):,}")
        elif filter_percentile < 90.0:
            self.logger.warning(f"⚠️  filter_percentile={filter_percentile} too aggressive (removes variance)")
            self.logger.warning(f"   Using filter_percentile=90 instead (remove bottom 10% only)")
            threshold = np.percentile(training_data[target_column], 90.0)
            original_len = len(training_data)
            training_data = training_data[training_data[target_column] >= threshold].copy()
            self.logger.info(f"   Kept samples: {len(training_data):,}")
        else:
            self.logger.info(f"\n✅ NO HARD FILTERING - using confidence weighting only")
        
        self.logger.info(f"   ✅ Training samples: {len(training_data):,}")
        
        if len(training_data) < 100:
            raise ValueError(f"Insufficient training data after filtering: {len(training_data)} samples")
        
        # Prepare data - CRITICAL FIX: Exclude ALL targets to prevent leakage
        exclude_cols = ['date', 'symbol', 'exchange', 'timeframe', 'sample_weight',
                       # Primary target
                       'quality_score',
                       # Sub-targets (components of quality_score) - MUST EXCLUDE!
                       'bounce_quality', 'hold_quality', 'trade_quality', 
                       'speed_quality', 'volume_confirmation_quality',
                       # Performance metrics (forward-looking targets) - MUST EXCLUDE!
                       'hit_rate', 'bounce_strength', 'max_bounce_strength',
                       'hold_strength', 'trade_profit', 'rejection_speed', 'volume_quality']
        
        # SAFER: Use only columns that explicitly start with 'feature_'
        feature_cols = [c for c in training_data.columns 
                       if c.startswith('feature_') and not pd.isna(training_data[c]).all()]
        
        self.logger.info(f"\n   🔒 TARGET LEAKAGE PREVENTION (HPO):")
        self.logger.info(f"      Using only 'feature_*' columns: {len(feature_cols)} features")
        
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found! All columns must start with 'feature_'")
        
        X = training_data[feature_cols].fillna(0.0)
        y = training_data[target_column]
        
        self.logger.info(f"   Initial features: {len(feature_cols)}")
        
        # ========================================
        # FIX 1: REMOVE CONSTANT FEATURES
        # ========================================
        self.logger.info(f"\n🔧 FIX 1: Removing constant/low-variance features...")
        feature_std = X.std()
        constant_features = feature_std[feature_std <= 0.001].index.tolist()
        low_var_features = feature_std[(feature_std > 0.001) & (feature_std < 0.01)].index.tolist()
        
        if constant_features:
            self.logger.info(f"   🚨 Removing {len(constant_features)} CONSTANT features (std ≤ 0.001):")
            for feat in constant_features[:10]:
                self.logger.info(f"      - {feat} (std={feature_std[feat]:.6f})")
            if len(constant_features) > 10:
                self.logger.info(f"      ... and {len(constant_features) - 10} more")
        
        if low_var_features:
            self.logger.info(f"   ⚠️  Found {len(low_var_features)} LOW-VARIANCE features (0.001 < std < 0.01) - keeping for now")
        
        # Remove constant features
        valid_features = feature_std[feature_std > 0.001].index.tolist()
        X = X[valid_features]
        
        self.logger.info(f"   ✅ Features after removing constants: {len(valid_features)} (removed {len(constant_features)})")
        
        # ========================================
        # FIX 2: FEATURE SELECTION (Top 50 by LGBM Importance)
        # ========================================
        if len(valid_features) > 50:
            self.logger.info(f"\n🔧 FIX 2: Selecting top 50 features using LightGBM importance...")
            
            # Train a quick model to get feature importance
            quick_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1,
                'seed': 42
            }
            
            train_data = lgb.Dataset(X, label=y)
            quick_model = lgb.train(
                quick_params,
                train_data,
                num_boost_round=50,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )
            
            # Get feature importance
            importance = quick_model.feature_importance(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Select top 50
            top_50_features = importance_df.head(50)['feature'].tolist()
            
            self.logger.info(f"   📊 Top 10 most important features:")
            for idx, row in importance_df.head(10).iterrows():
                self.logger.info(f"      {row['feature']:<50} importance={row['importance']:>10.0f}")
            
            self.logger.info(f"   ✅ Selected top 50 features (from {len(valid_features)})")
        
            # Keep only top 50
            X = X[top_50_features]
            self.feature_names = top_50_features
        else:
            self.feature_names = valid_features
            self.logger.info(f"   ℹ️  Only {len(valid_features)} features - keeping all (no selection needed)")
        
        self.logger.info(f"   ✅ Final feature count: {len(self.feature_names)}")
        
        # ========================================
        # FIX 3: ENHANCED ANTI-OVERFITTING SEARCH SPACE
        # ========================================
        # EXPANDED regularization ranges based on analysis feedback
        # Focus on preventing overfitting with small sample sizes
        search_space = {
            # Model complexity (prioritize simplicity)
            # Conservative range to prevent overfitting on small datasets
            'num_leaves': {'type': 'int', 'low': 10, 'high': 40, 'default': 23},  # Even more conservative
            'max_depth': {'type': 'int', 'low': 3, 'high': 6, 'default': 5},      # Reduced max from 7
            
            # EXPANDED REGULARIZATION RANGE - MANDATORY (min > 0)
            # Allow much stronger regularization for high variance scenarios
            'lambda_l1': {'type': 'float', 'low': 0.5, 'high': 50.0, 'default': 2.0, 'log': True},  # Expanded from 20
            'lambda_l2': {'type': 'float', 'low': 0.5, 'high': 50.0, 'default': 2.0, 'log': True},  # Expanded from 20
            
            # Require EVEN MORE evidence per leaf (prevent memorization)
            'min_data_in_leaf': {'type': 'int', 'low': 20, 'high': 200, 'default': 60},  # Expanded upper bound
            
            # SLOWER learning rate (more stable, less overfitting)
            'learning_rate': {'type': 'float', 'low': 0.003, 'high': 0.05, 'default': 0.01, 'log': True},  # Expanded range
            
            # Prevent weak splits that overfit noise
            'min_gain_to_split': {'type': 'float', 'low': 0.1, 'high': 5.0, 'default': 0.5},  # Expanded range
            
            # AGGRESSIVE subsampling to reduce overfitting
            'feature_fraction': {'type': 'float', 'low': 0.4, 'high': 0.85, 'default': 0.6},  # Expanded lower bound
            'bagging_fraction': {'type': 'float', 'low': 0.4, 'high': 0.85, 'default': 0.6},  # Expanded lower bound
            'bagging_freq': {'type': 'int', 'low': 1, 'high': 15, 'default': 5},  # Expanded upper bound
        }
        
        self.logger.info(f"\n🔧 FIX 3: ENHANCED ANTI-OVERFITTING HPO Configuration:")
        self.logger.info(f"   ✅ EXPANDED L1 regularization: 0.5 → 50.0 (was 20.0)")
        self.logger.info(f"   ✅ EXPANDED L2 regularization: 0.5 → 50.0 (was 20.0)")
        self.logger.info(f"   ✅ Min samples per leaf: 20-200 (expanded from 30-150)")
        self.logger.info(f"   ✅ Max tree depth: 3-6 (reduced from 3-7)")
        self.logger.info(f"   ✅ Learning rate: 0.003-0.05 (expanded range)")
        self.logger.info(f"   ✅ Aggressive subsampling: 40-85% (expanded range)")
        self.logger.info(f"   ℹ️  These expanded ranges allow HPO to find optimal regularization for high-variance data")
        
        # Create model factory for HPO
        def lgbm_factory(**params):
            from sklearn.base import BaseEstimator, RegressorMixin
            
            class LGBMWrapper(BaseEstimator, RegressorMixin):
                def __init__(self, **lgbm_params):
                    self.lgbm_params = lgbm_params
                    self.model_ = None
                
                def fit(self, X, y):
                    full_params = {
                        'objective': 'regression',
                        'metric': 'rmse',
                        'boosting_type': 'gbdt',
                        'bagging_freq': 5,
                        'verbose': -1,
                        'seed': 42,
                        'force_col_wise': True,
                        **self.lgbm_params
                    }
                    
                    train_data = lgb.Dataset(X, label=y)
                    self.model_ = lgb.train(
                        full_params,
                        train_data,
                        num_boost_round=100,  # Limited for HPO speed
                        valid_sets=[train_data],
                        callbacks=[lgb.early_stopping(20, verbose=False)]
                    )
                    return self
                
                def predict(self, X):
                    return self.model_.predict(X)
            
            return LGBMWrapper(**params)
        
        # Run HPO
        self.logger.info(f"🚀 Starting {method} HPO optimization...")
        
        hpo_results = optimize_hyperparameters(
            model_factory=lgbm_factory,
            X=X.values,
            y=y.values,
            search_space=search_space,
            n_trials=n_trials,
            method=method,
            scoring='neg_mean_squared_error',  # Minimize MSE
            cv=TimeSeriesSplit(n_splits=n_folds)
        )
        
        if 'error' in hpo_results:
            self.logger.error(f"❌ HPO failed: {hpo_results['error']}")
            self.logger.warning("⚠️ Falling back to default anti-overfitting config")
            best_params = self._get_default_config()
        else:
            best_params = hpo_results.get('best_params', {})
            best_score = hpo_results.get('best_score', 0)
            self.logger.info(f"✅ HPO completed: Best CV score = {best_score:.4f}")
            self.logger.info(f"🏆 Best parameters found:")
            for param, value in best_params.items():
                self.logger.info(f"   {param}: {value}")
        
        # Update config with best parameters
        self.config.update(best_params)
        self.config.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'force_col_wise': True
        })
        
        # Train final model with best parameters
        self.logger.info(f"🎓 Training final model with optimized parameters...")
        final_results = self.train(
            training_data,
            target_column,
            n_folds,
            num_boost_round,
            early_stopping_rounds
        )
        
        # Add HPO results to final results
        final_results['hpo_results'] = hpo_results
        final_results['hpo_best_params'] = best_params
        
        return final_results
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict quality scores for SR levels.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of quality scores (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first or load() a trained model.")
        
        # Ensure feature order matches training
        try:
            X = features[self.feature_names]
        except KeyError as e:
            missing = set(self.feature_names) - set(features.columns)
            self.logger.error(f"Missing features: {missing}")
            raise ValueError(f"Missing features: {missing}")
        
        # Fill NaN values
        X = X.fillna(0.0)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Clip to [0, 1] range (quality scores)
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def predict_single(self, features_dict: Dict[str, float]) -> float:
        """Predict quality for a single SR level.
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            Quality score (0-1)
        """
        features_df = pd.DataFrame([features_dict])
        predictions = self.predict(features_df)
        return float(predictions[0])
    
    def _log_feature_importance(self):
        """Log top 20 most important features."""
        if self.model is None:
            return
        
        importance = self.model.feature_importance(importance_type='gain')
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('importance', ascending=False)
        
        self.logger.info("\n🏆 Top 20 Feature Importance:")
        for idx, row in feature_importance_df.head(20).iterrows():
            self.logger.info(f"   {row['feature']:<35} {row['importance']:>8.0f} ({row['importance_pct']:>5.1f}%)")
        
        # Return for analysis
        return feature_importance_df
    
    def evaluate_ranking(self, X_test: pd.DataFrame, y_true: pd.Series, 
                        k: int = 10, quality_threshold: float = 0.7) -> Dict:
        """Evaluate model as a RANKING system (information retrieval metrics).
        
        This is what actually matters for SR detection!
        Users look at TOP K levels, not all levels.
        
        Args:
            X_test: Test features
            y_true: True quality scores
            k: Number of top levels to evaluate (default: 10)
            quality_threshold: Threshold for "good" level (default: 0.7)
            
        Returns:
            Dictionary with ranking metrics:
            - precision_at_k: Of top K, how many are actually good?
            - spearman_rho: Rank correlation (-1 to 1)
            - ndcg_at_k: Normalized discounted cumulative gain
            - r2_score: Traditional regression metric (for comparison)
        """
        if self.model is None:
            raise ValueError("No trained model. Train model first.")
        
        # Predict using self.predict() to ensure proper feature selection
        y_pred = self.predict(X_test)
        
        # Metric 1: Precision @ K (most important!)
        precision_k = self._calculate_precision_at_k(
            y_pred, y_true, k=k, threshold=quality_threshold
        )
        
        # Metric 2: Spearman rank correlation
        spearman_rho, p_value = spearmanr(y_pred, y_true)
        
        # Metric 3: NDCG @ K
        ndcg_k = self._calculate_ndcg_at_k(y_pred, y_true, k=k)
        
        # Metric 4: Traditional R² (for reference)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        results = {
            'precision_at_k': precision_k,
            'spearman_rho': spearman_rho,
            'spearman_p_value': p_value,
            'ndcg_at_k': ndcg_k,
            'r2_score': r2,
            'rmse': rmse,
            'k': k,
            'quality_threshold': quality_threshold,
            'total_samples': len(y_true)
        }
        
        # Log results
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"  RANKING EVALUATION (Top {k} Levels)")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"\n📊 RANKING METRICS (What Matters!):")
        self.logger.info(f"   Precision@{k}:     {precision_k*100:.1f}% "
                        f"({int(precision_k*k)}/{k} are good)")
        self.logger.info(f"   Spearman ρ:       {spearman_rho:.3f} "
                        f"(p={p_value:.4f})")
        self.logger.info(f"   NDCG@{k}:         {ndcg_k:.3f}")
        
        self.logger.info(f"\n📈 REGRESSION METRICS (For Reference):")
        self.logger.info(f"   R² Score:         {r2:.3f}")
        self.logger.info(f"   RMSE:             {rmse:.3f}")
        
        # Interpretation
        self.logger.info(f"\n💡 INTERPRETATION:")
        
        if precision_k >= 0.8:
            self.logger.info(f"   ✅ Excellent: {precision_k*100:.0f}% of top {k} are strong!")
        elif precision_k >= 0.6:
            self.logger.info(f"   🟡 Good: {precision_k*100:.0f}% of top {k} are strong")
        else:
            self.logger.info(f"   ❌ Poor: Only {precision_k*100:.0f}% of top {k} are strong")
        
        if spearman_rho >= 0.7:
            self.logger.info(f"   ✅ Strong ranking correlation")
        elif spearman_rho >= 0.5:
            self.logger.info(f"   🟡 Moderate ranking correlation")
        else:
            self.logger.info(f"   ❌ Weak ranking correlation")
        
        self.logger.info(f"{'='*70}\n")
        
        return results
    
    def _calculate_precision_at_k(self, y_pred: np.ndarray, y_true: np.ndarray,
                                  k: int, threshold: float) -> float:
        """Calculate Precision@K.
        
        Of the top K predicted levels, how many are actually good?
        """
        # Get indices of top K predictions
        top_k_indices = np.argsort(y_pred)[-k:][::-1]
        
        # Count how many are actually good
        good_count = np.sum(y_true[top_k_indices] >= threshold)
        
        return good_count / k
    
    def _calculate_ndcg_at_k(self, y_pred: np.ndarray, y_true: np.ndarray,
                            k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain @ K.
        
        Measures ranking quality with position-based weighting.
        Getting position #1 wrong hurts more than getting #10 wrong.
        """
        # Get top K by prediction
        top_k_indices = np.argsort(y_pred)[-k:][::-1]
        
        # DCG: Discounted Cumulative Gain
        dcg = 0
        for i, idx in enumerate(top_k_indices):
            relevance = y_true[idx]
            position = i + 1
            dcg += relevance / np.log2(position + 1)
        
        # IDCG: Ideal DCG (perfect ranking)
        ideal_indices = np.argsort(y_true)[-k:][::-1]
        idcg = 0
        for i, idx in enumerate(ideal_indices):
            relevance = y_true[idx]
            position = i + 1
            idcg += relevance / np.log2(position + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def save(self, path: str):
        """Save trained model and metadata.
        
        Args:
            path: Path to save model (e.g., 'models/sr_quality_model.lgb')
        """
        if self.model is None:
            raise ValueError("No model to save! Train first.")
        
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'config': self.config,
            'saved_at': datetime.now().isoformat()
        }
        
        metadata_path = str(model_path) + '.metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"✅ Model saved to {model_path}")
        self.logger.info(f"✅ Metadata saved to {metadata_path}")
    
    def load(self, path: str):
        """Load trained model and metadata.
        
        Args:
            path: Path to model file
        """
        model_path = Path(path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        # Load LightGBM model
        self.model = lgb.Booster(model_file=str(model_path))
        
        # Load metadata
        metadata_path = str(model_path) + '.metadata.json'
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.training_metrics = metadata.get('training_metrics', {})
            self.config = metadata.get('config', self._get_default_config())
        else:
            self.logger.warning(f"⚠️ Metadata not found, using default config")
        
        self.logger.info(f"✅ Model loaded from {path}")
        self.logger.info(f"   Features: {len(self.feature_names)}")
        if 'avg_metrics' in self.training_metrics:
            avg_r2 = self.training_metrics['avg_metrics'].get('avg_val_r2', 'N/A')
            self.logger.info(f"   Avg Val R²: {avg_r2}")


# Convenience functions
def train_sr_quality_model(training_data_path: str, 
                          output_model_path: str = 'models/sr_quality_model.lgb') -> SRQualityModel:
    """Train and save SR quality model.
    
    Args:
        training_data_path: Path to training data parquet
        output_model_path: Where to save trained model
        
    Returns:
        Trained model
    """
    logger.info(f"🚀 Training SR Quality Model")
    logger.info(f"   Training data: {training_data_path}")
    logger.info(f"   Output model: {output_model_path}")
    
    # Load training data
    training_df = pd.read_parquet(training_data_path)
    logger.info(f"   Loaded {len(training_df)} training samples")
    
    # Create and train model
    model = SRQualityModel()
    metrics = model.train(training_df)
    
    # Save model
    model.save(output_model_path)
    
    logger.info(f"✅ Model training complete!")
    
    return model


def load_sr_quality_model(model_path: str = 'models/sr_quality_model.lgb') -> SRQualityModel:
    """Load trained SR quality model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model ready for predictions
    """
    model = SRQualityModel()
    model.load(model_path)
    return model

