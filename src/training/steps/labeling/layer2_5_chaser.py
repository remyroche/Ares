"""
Layer 2.5: The Chaser - Non-Linear Alpha Extraction System

The Chaser is the "High-ROI Muscle" that hunts for non-linear alpha
in the gaps of market physics, operating on causal residuals.

Key Components:
1. Causal Residual Targeting (y~ = y_actual - y_causal_anchor)
2. Non-Causal Feature Selection (technical indicators only) - Using Huber Regressor
3. Independent XGBoost + CatBoost + ExtraTrees + LightGBM Models
4. Enhanced Model Comparison and Ranking
5. Conflict Detection with Causal Anchor
6. Confidence Scoring for Meta-Learner
"""

import time
import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

import warnings
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs

class Layer25Chaser:
    """
    Layer 2.5 Chaser: Non-linear alpha extraction from causal residuals.
    
    The Chaser learns only the "unexplained alpha" that the Causal Anchor
    cannot capture, focusing on temporary inefficiencies and anomalies.
    """
    
    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        cat_params: Optional[Dict] = None,
        et_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        confidence_threshold: float = 0.5,
        conflict_threshold: float = 2.0,
        verbose: bool = True
    ):
        """
        Initialize Layer 2.5 Chaser.

        Args:
            xgb_params: XGBoost hyperparameters
            cat_params: CatBoost hyperparameters
            et_params: ExtraTrees hyperparameters
            lgb_params: LightGBM hyperparameters
            confidence_threshold: Minimum confidence for predictions
            conflict_threshold: Threshold for conflict detection (std deviations)
            verbose: Whether to print progress information
        """
        self.verbose = verbose

        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.6,
            'colsample_bytree': 0.8,
            'colsample_bynode': 0.4,
            'num_parallel_tree': 7,
            'min_child_weight': 10,
            'gamma': 1.1,
            'reg_lambda': 50,  # L2 regularization
            'random_state': 42,
            'n_jobs': -1
        }

        # Default CatBoost parameters
        self.cat_params = cat_params or {
            'iterations': 200,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 20,
            'random_strength': 5,
            'subsample': 0.6,
            'colsample_bylevel': 0.5,
            'leaf_estimation_iterations': 10,
            'bootstrap_type': 'MVS',
            'random_seed': 42,
            'verbose': False,
            'od_type': 'Iter',
            'od_wait': 20
        }

        # Default ExtraTrees parameters
        self.et_params = et_params or {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }

        # Default LightGBM parameters
        self.lgb_params = lgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.03,
            'num_leaves': 31,
            'path_smooth': 20,
            'reg_lambda': 10,
            'extra_trees': True,
            'linear_tree': True,
            'min_gain_to_split': 0.02,
            'bagging_fraction': 0.7,
            'feature_fraction': 0.6,
            'lambda_l1': 1.0,
            'max_bin': 63,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.conflict_threshold = conflict_threshold

        # Initialize models
        self.xgb_model = None
        self.cat_model = None
        self.et_model = None
        self.lgb_model = None

        # Huber Teacher components
        self.huber_model = None
        self.scaler = None
        
        # Training metadata
        self.feature_names = None
        self.training_score = None
        self.cv_scores = None
        self.prediction_std = None
        self.pruned_features = []
        self.constraints = {}
        self.interaction_constraints = None
        
    def _validate_inputs(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series,
        handle_outliers: bool = False,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
        outlier_handling_strategy: str = 'winsorize'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and prepare input data with optional outlier handling."""
        try:
            # Remove NaN values
            valid_mask = ~(X_non_causal.isna().any(axis=1) | y_residuals.isna())
            X_clean = X_non_causal[valid_mask]
            y_clean = y_residuals[valid_mask]

            if len(X_clean) < 100:
                raise ValueError(f"Insufficient training data: {len(X_clean)} samples")

            # Handle outliers if requested
            if handle_outliers:
                y_clean, outlier_mask = self.detect_and_handle_outliers(
                    y_clean,
                    method=outlier_method,
                    threshold=outlier_threshold,
                    handling_strategy=outlier_handling_strategy
                )
                # Remove outliers if strategy is 'remove'
                if outlier_handling_strategy == 'remove':
                    X_clean = X_clean[~outlier_mask]

            if self.verbose:
                tprint_info(f"🔍 Chaser data validation:")
                nan_removed = len(X_non_causal) - len(X_non_causal[valid_mask])
                final_samples = len(X_clean)
                tprint_info(f"   - Samples: {final_samples} (NaN removed: {nan_removed})")
                if handle_outliers and outlier_handling_strategy == 'remove':
                    outliers_removed = outlier_mask.sum()
                    tprint_info(f"   - Outliers removed: {outliers_removed}")
                tprint_info(f"   - Features: {len(X_clean.columns)}")
                tprint_info(f"   - Target mean: {y_clean.mean():.6f}")
                tprint_info(f"   - Target std: {y_clean.std():.6f}")

            return X_clean, y_clean

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Input validation failed: {e}")
            raise
    
    def fit(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series,
        cv_folds: int = 5,
        early_stopping_rounds: int = 50,
        handle_outliers: bool = False,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
        outlier_handling_strategy: str = 'winsorize'
    ) -> Dict[str, Any]:
        """
        Fit the Chaser models independently on causal residuals using Huber Teacher.

        Args:
            X_non_causal: Non-causal features
            y_residuals: Causal residuals
            cv_folds: Number of cross-validation folds
            early_stopping_rounds: Early stopping patience
            handle_outliers: Whether to detect and handle outliers
            outlier_method: Outlier detection method
            outlier_threshold: Threshold for outlier detection
            outlier_handling_strategy: Strategy to handle outliers

        Returns:
            Dictionary with training metrics
        """
        try:
            if self.verbose:
                tprint_info("🚀 Training Layer 2.5 Chaser on Causal Residuals...")

            # Validate inputs
            X_clean, y_clean = self._validate_inputs(
                X_non_causal, y_residuals,
                handle_outliers=handle_outliers,
                outlier_method=outlier_method,
                outlier_threshold=outlier_threshold,
                outlier_handling_strategy=outlier_handling_strategy
            )

            # --- HUBER TEACHER ---
            tprint_info("   🧑‍🏫 Running Huber Teacher for Feature Selection, Constraints & Warm Start...")
            teacher_outputs = prepare_huber_teacher_outputs(X_clean, y_clean)

            self.huber_model = teacher_outputs['huber_model']
            self.scaler = teacher_outputs['scaler']
            self.feature_names = teacher_outputs['selected_features']
            self.interaction_constraints = teacher_outputs['interaction_constraints']

            # Identify pruned features
            original_features = X_clean.columns.tolist()
            self.pruned_features = [f for f in original_features if f not in self.feature_names]
            tprint_info(f"      → Pruned {len(self.pruned_features)} features, {len(self.feature_names)} remaining.")
            
            # Map constraints
            self.constraints = dict(zip(self.feature_names, teacher_outputs['monotonic_constraints']))
            n_inc = sum(1 for x in teacher_outputs['monotonic_constraints'] if x == 1)
            n_dec = sum(1 for x in teacher_outputs['monotonic_constraints'] if x == -1)
            tprint_info(f"      → Constraints: {n_inc} Increasing, {n_dec} Decreasing.")

            # Filter data to selected features
            X_selected = X_clean[self.feature_names]
            
            # Get full warm start (for final fit)
            full_warm_start = teacher_outputs['warm_start']['train']

            # --- MANUAL CV FOR WARM START ---
            tscv = TimeSeriesSplit(n_splits=cv_folds)

            xgb_scores, cat_scores, et_scores, lgb_scores = [], [], [], []

            if self.verbose:
                tprint_info("   📊 Running Cross-Validation with Warm Start...")

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
                X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_tr, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                
                # Get warm start for this fold (subsetting the full prediction is valid here since
                # Huber was trained on full data as a 'Teacher' / 'Prior'.
                # Note: This introduces some leakage from Huber to CV, but accepted for 'Teacher' paradigm)
                ws_tr = full_warm_start[train_idx]
                ws_val = full_warm_start[val_idx]
                
                # 1. XGBoost
                curr_xgb = self.xgb_params.copy()
                curr_xgb['monotone_constraints'] = teacher_outputs['monotonic_constraints']
                if self.interaction_constraints:
                    curr_xgb['interaction_constraints'] = self.interaction_constraints

                xgb_m = xgb.XGBRegressor(**curr_xgb)
                xgb_m.fit(X_tr, y_tr, base_margin=ws_tr)
                # Predict adding margin
                p_val = xgb_m.predict(X_val, base_margin=ws_val)
                xgb_scores.append(mean_squared_error(y_val, p_val))

                # 2. CatBoost
                if CATBOOST_AVAILABLE:
                    curr_cat = self.cat_params.copy()
                    curr_cat['monotone_constraints'] = self.constraints
                    # CatBoost doesn't support interaction constraints explicitly in the same way or via simple param

                    cat_m = cb.CatBoostRegressor(**curr_cat)
                    cat_m.fit(X_tr, y_tr, baseline=ws_tr)
                    # CatBoost predict returns raw formula, need to add baseline manually
                    p_val_raw = cat_m.predict(X_val)
                    p_val = p_val_raw + ws_val
                    cat_scores.append(mean_squared_error(y_val, p_val))

                # 3. LightGBM
                if LGBM_AVAILABLE:
                    curr_lgb = self.lgb_params.copy()
                    curr_lgb['monotone_constraints'] = list(teacher_outputs['monotonic_constraints'])
                    if self.interaction_constraints:
                        curr_lgb['interaction_constraints'] = self.interaction_constraints

                    lgb_m = lgb.LGBMRegressor(**curr_lgb)
                    lgb_m.fit(X_tr, y_tr, init_score=ws_tr)
                    # LGBM predict returns margin-adjusted if raw_score=False, wait.
                    # Documentation: "If init_score was used in training, it is NOT automatically added to prediction"
                    p_val_raw = lgb_m.predict(X_val)
                    p_val = p_val_raw + ws_val
                    lgb_scores.append(mean_squared_error(y_val, p_val))

                # 4. ExtraTrees (No Warm Start)
                curr_et = self.et_params.copy()
                # Try adding monotonic constraints (sklearn 1.4+)
                try:
                    curr_et['monotonic_cst'] = teacher_outputs['monotonic_constraints']
                    et_m = ExtraTreesRegressor(**curr_et)
                    et_m.fit(X_tr, y_tr)
                except TypeError:
                    # Fallback for older sklearn
                    if 'monotonic_cst' in curr_et:
                        del curr_et['monotonic_cst']
                    et_m = ExtraTreesRegressor(**curr_et)
                    et_m.fit(X_tr, y_tr)

                p_val = et_m.predict(X_val)
                et_scores.append(mean_squared_error(y_val, p_val))

            # Store CV Scores
            self.cv_scores = {
                'xgb_cv_mse': np.mean(xgb_scores),
                'xgb_cv_std': np.std(xgb_scores),
                'cat_cv_mse': np.mean(cat_scores) if cat_scores else 0.0,
                'cat_cv_std': np.std(cat_scores) if cat_scores else 0.0,
                'lgb_cv_mse': np.mean(lgb_scores) if lgb_scores else 0.0,
                'lgb_cv_std': np.std(lgb_scores) if lgb_scores else 0.0,
                'et_cv_mse': np.mean(et_scores),
                'et_cv_std': np.std(et_scores)
            }

            # --- FINAL TRAINING ---
            if self.verbose:
                tprint_info("   🏁 Fitting final models on full data...")

            # XGBoost
            final_xgb_params = self.xgb_params.copy()
            final_xgb_params['monotone_constraints'] = teacher_outputs['monotonic_constraints']
            if self.interaction_constraints:
                final_xgb_params['interaction_constraints'] = self.interaction_constraints
            self.xgb_model = xgb.XGBRegressor(**final_xgb_params)
            self.xgb_model.fit(X_selected, y_clean, base_margin=full_warm_start)

            # CatBoost
            if CATBOOST_AVAILABLE:
                final_cat_params = self.cat_params.copy()
                final_cat_params['monotone_constraints'] = self.constraints
                self.cat_model = cb.CatBoostRegressor(**final_cat_params)
                self.cat_model.fit(X_selected, y_clean, baseline=full_warm_start)

            # LightGBM
            if LGBM_AVAILABLE:
                final_lgb_params = self.lgb_params.copy()
                final_lgb_params['monotone_constraints'] = list(teacher_outputs['monotonic_constraints'])
                if self.interaction_constraints:
                    final_lgb_params['interaction_constraints'] = self.interaction_constraints
                self.lgb_model = lgb.LGBMRegressor(**final_lgb_params)
                self.lgb_model.fit(X_selected, y_clean, init_score=full_warm_start)

            # ExtraTrees
            final_et_params = self.et_params.copy()
            try:
                final_et_params['monotonic_cst'] = teacher_outputs['monotonic_constraints']
                self.et_model = ExtraTreesRegressor(**final_et_params)
                self.et_model.fit(X_selected, y_clean)
            except TypeError:
                if 'monotonic_cst' in final_et_params:
                    del final_et_params['monotonic_cst']
                self.et_model = ExtraTreesRegressor(**final_et_params)
                self.et_model.fit(X_selected, y_clean)

            # Training Metrics
            xgb_pred = self.xgb_model.predict(X_selected, base_margin=full_warm_start)
            et_pred = self.et_model.predict(X_selected)
            cat_pred = (self.cat_model.predict(X_selected) + full_warm_start) if self.cat_model else xgb_pred
            lgb_pred = (self.lgb_model.predict(X_selected) + full_warm_start) if self.lgb_model else xgb_pred

            self.training_score = {
                'xgb': {'rmse': np.sqrt(mean_squared_error(y_clean, xgb_pred))},
                'cat': {'rmse': np.sqrt(mean_squared_error(y_clean, cat_pred))},
                'lgb': {'rmse': np.sqrt(mean_squared_error(y_clean, lgb_pred))},
                'et': {'rmse': np.sqrt(mean_squared_error(y_clean, et_pred))}
            }
            
            # Prediction STD for confidence
            all_preds = [xgb_pred, et_pred]
            if self.cat_model: all_preds.append(cat_pred)
            if self.lgb_model: all_preds.append(lgb_pred)
            self.prediction_std = np.std(np.column_stack(all_preds), axis=1).mean()

            if self.verbose:
                tprint_success("✅ Chaser training complete!")
                tprint_info(f"   - XGBoost CV RMSE: {np.sqrt(self.cv_scores['xgb_cv_mse']):.6f}")
                if self.cat_model: tprint_info(f"   - CatBoost CV RMSE: {np.sqrt(self.cv_scores['cat_cv_mse']):.6f}")
                if self.lgb_model: tprint_info(f"   - LightGBM CV RMSE: {np.sqrt(self.cv_scores['lgb_cv_mse']):.6f}")
                tprint_info(f"   - ExtraTrees CV RMSE: {np.sqrt(self.cv_scores['et_cv_mse']):.6f}")

            return {
                'training_metrics': self.training_score,
                'cv_metrics': self.cv_scores,
                'feature_count': len(self.feature_names),
                'pruned_features': len(self.pruned_features)
            }

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Chaser training failed: {e}")
            raise
    
    def predict(
        self,
        X_non_causal: pd.DataFrame,
        return_confidence: bool = True
    ) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], np.ndarray]]:
        """
        Predict residual alpha using all Chaser models.

        Args:
            X_non_causal: Non-causal features
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary with predictions and optionally confidence scores
        """
        try:
            if self.xgb_model is None:
                raise ValueError("Chaser models not fitted. Call fit() first.")

            # --- Huber Prediction ---
            # Huber teacher uses ALL features (robustly scaled)
            # We must pass the full feature set to the scaler, matching fit time
            huber_pred = self.huber_model.predict(
                self.scaler.transform(X_non_causal.fillna(0))
            )

            # --- Tree Predictions ---
            # Tree models use PRUNED features (self.feature_names)
            if self.feature_names is not None:
                X_aligned = X_non_causal[self.feature_names].fillna(0)
            else:
                X_aligned = X_non_causal.fillna(0)

            # XGBoost (add margin)
            xgb_pred = self.xgb_model.predict(X_aligned, base_margin=huber_pred)

            # CatBoost (add baseline manually)
            if self.cat_model:
                cat_pred = self.cat_model.predict(X_aligned) + huber_pred
            else:
                cat_pred = xgb_pred
            
            # LightGBM (add init_score manually)
            if self.lgb_model:
                lgb_pred = self.lgb_model.predict(X_aligned) + huber_pred
            else:
                lgb_pred = xgb_pred
                
            # ExtraTrees (No warm start)
            et_pred = self.et_model.predict(X_aligned)

            predictions = {
                'xgb': xgb_pred,
                'cat': cat_pred,
                'lgb': lgb_pred,
                'et': et_pred
            }

            if not return_confidence:
                return predictions

            # Calculate confidence
            all_preds_list = [xgb_pred, et_pred]
            if self.cat_model: all_preds_list.append(cat_pred)
            if self.lgb_model: all_preds_list.append(lgb_pred)

            all_preds = np.column_stack(all_preds_list)
            pred_std = np.std(all_preds, axis=1)
            confidence = 1.0 / (1.0 + pred_std / (self.prediction_std + 1e-8))

            return predictions, confidence

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Chaser prediction failed: {e}")
            raise
    
    def optimize_hyperparameters(
        self,
        X_non_causal: pd.DataFrame,
        y_residuals: pd.Series,
        optimization_fraction: float = 0.3,
        n_trials: int = 50,
        timeout: int = 3600,
        cv_folds: int = 3,
        random_state: int = 42
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for XGBoost, CatBoost, and LightGBM using Optuna with Huber Warm Start.
        """
        try:
            if self.verbose:
                tprint_info("🔬 Starting hyperparameter optimization with Optuna...")

            # Use subset of data for optimization
            if optimization_fraction < 1.0:
                X_opt, _, y_opt, _ = train_test_split(
                    X_non_causal, y_residuals,
                    train_size=optimization_fraction,
                    random_state=random_state,
                    shuffle=False
                )
            else:
                X_opt, y_opt = X_non_causal, y_residuals

            # --- Huber Teacher for Optimization Subset ---
            # We run teacher on the optimization subset to get correct base margins for CV
            teacher_outputs = prepare_huber_teacher_outputs(X_opt, y_opt)
            X_opt_sel = X_opt[teacher_outputs['selected_features']]
            full_ws = teacher_outputs['warm_start']['train']
            monotonic = teacher_outputs['monotonic_constraints']
            interactions = teacher_outputs['interaction_constraints']

            tscv = TimeSeriesSplit(n_splits=cv_folds)

            # --- XGBoost Optimization ---
            if self.verbose: tprint_info("   🚀 Optimizing XGBoost...")

            def xgb_objective(trial):
                params = {
                    'n_estimators': 200, # Fixed as per base
                    'max_depth': trial.suggest_int('max_depth', 4, 6),
                    'learning_rate': 0.03, # Fixed base
                    'subsample': 0.6,
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 0.5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
                    'gamma': trial.suggest_float('gamma', 0.5, 2.0),
                    'reg_lambda': 50, # Fixed
                    'num_parallel_tree': 7,
                    'random_state': random_state,
                    'n_jobs': -1,
                    'monotone_constraints': monotonic
                }
                if interactions: params['interaction_constraints'] = interactions

                # Manual CV loop
                scores = []
                for step, (tr_idx, val_idx) in enumerate(tscv.split(X_opt_sel)):
                    X_tr, X_val = X_opt_sel.iloc[tr_idx], X_opt_sel.iloc[val_idx]
                    y_tr, y_val = y_opt.iloc[tr_idx], y_opt.iloc[val_idx]
                    ws_tr, ws_val = full_ws[tr_idx], full_ws[val_idx]

                    m = xgb.XGBRegressor(**params)
                    m.fit(X_tr, y_tr, base_margin=ws_tr)
                    p = m.predict(X_val, base_margin=ws_val)
                    mse = mean_squared_error(y_val, p)
                    scores.append(mse)

                    # Report intermediate result for pruning
                    # We report the mean score so far to be more robust
                    current_mean_mse = np.mean(scores)
                    trial.report(-current_mean_mse, step=step)
                    if trial.should_prune(): raise optuna.TrialPruned()

                mean_score = np.mean(scores)
                return -mean_score

            xgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state), pruner=MedianPruner())
            xgb_study.optimize(xgb_objective, n_trials=n_trials)
            self.xgb_params.update(xgb_study.best_params)

            # --- LightGBM Optimization ---
            if LGBM_AVAILABLE:
                if self.verbose: tprint_info("   🚀 Optimizing LightGBM...")

                def lgb_objective(trial):
                    params = {
                        'n_estimators': 200,
                        'max_depth': 6, # Fixed base or can vary
                        'learning_rate': 0.03,
                        'num_leaves': 31,
                        'path_smooth': 20,
                        'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0),
                        'extra_trees': True,
                        'linear_tree': True,
                        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.05),
                        'bagging_fraction': 0.7,
                        'feature_fraction': 0.6,
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 5.0),
                        'max_bin': 63,
                        'monotone_constraints': list(monotonic),
                        'random_state': random_state,
                        'n_jobs': -1,
                        'verbose': -1
                    }
                    if interactions: params['interaction_constraints'] = interactions

                    scores = []
                    for step, (tr_idx, val_idx) in enumerate(tscv.split(X_opt_sel)):
                        X_tr, X_val = X_opt_sel.iloc[tr_idx], X_opt_sel.iloc[val_idx]
                        y_tr, y_val = y_opt.iloc[tr_idx], y_opt.iloc[val_idx]
                        ws_tr, ws_val = full_ws[tr_idx], full_ws[val_idx]

                        m = lgb.LGBMRegressor(**params)
                        m.fit(X_tr, y_tr, init_score=ws_tr)
                        p = m.predict(X_val) + ws_val
                        mse = mean_squared_error(y_val, p)
                        scores.append(mse)

                        current_mean_mse = np.mean(scores)
                        trial.report(-current_mean_mse, step=step)
                        if trial.should_prune(): raise optuna.TrialPruned()

                    mean_score = np.mean(scores)
                    return -mean_score

                lgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state), pruner=MedianPruner())
                lgb_study.optimize(lgb_objective, n_trials=n_trials)
                self.lgb_params.update(lgb_study.best_params)

            # --- CatBoost Optimization ---
            # (Keeping existing logic mostly but updated params)
            if self.verbose: tprint_info("   🚀 Optimizing CatBoost...")

            def cat_objective(trial):
                # Ensure monotonic constraints map to correct feature names
                # X_opt_sel has columns matching 'selected_features' from Huber teacher
                # monotonic tuple matches X_opt_sel column order
                mono_dict = dict(zip(X_opt_sel.columns, monotonic))

                params = {
                    'iterations': 200,
                    'depth': 6,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 20,
                    'random_strength': 5,
                    'subsample': 0.6,
                    'colsample_bylevel': 0.5,
                    'leaf_estimation_iterations': 10,
                    'bootstrap_type': 'MVS',
                    # Optimization target?
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                    'random_seed': random_state,
                    'verbose': False,
                    'monotone_constraints': mono_dict
                }

                scores = []
                for step, (tr_idx, val_idx) in enumerate(tscv.split(X_opt_sel)):
                    X_tr, X_val = X_opt_sel.iloc[tr_idx], X_opt_sel.iloc[val_idx]
                    y_tr, y_val = y_opt.iloc[tr_idx], y_opt.iloc[val_idx]
                    ws_tr, ws_val = full_ws[tr_idx], full_ws[val_idx]

                    m = cb.CatBoostRegressor(**params)
                    # CatBoost needs dataframe to match feature names in constraints
                    m.fit(X_tr, y_tr, baseline=ws_tr)
                    p = m.predict(X_val) + ws_val
                    mse = mean_squared_error(y_val, p)
                    scores.append(mse)

                    current_mean_mse = np.mean(scores)
                    trial.report(-current_mean_mse, step=step)
                    if trial.should_prune(): raise optuna.TrialPruned()

                mean_score = np.mean(scores)
                return -mean_score

            if CATBOOST_AVAILABLE:
                cat_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=random_state), pruner=MedianPruner())
                cat_study.optimize(cat_objective, n_trials=n_trials)
                self.cat_params.update(cat_study.best_params)

            return {
                'xgb': self.xgb_params,
                'lgb': self.lgb_params,
                'cat': self.cat_params
            }

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Hyperparameter optimization failed: {e}")
            raise

    # ... Helper methods like detect_conflict, get_feature_importance, evaluate ...
    # Reusing existing methods where applicable, updating them if they reference models

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from all models."""
        try:
            if self.xgb_model is None: raise ValueError("Models not fitted")

            xgb_imp = dict(zip(self.feature_names, self.xgb_model.feature_importances_))
            et_imp = dict(zip(self.feature_names, self.et_model.feature_importances_))

            cat_imp = xgb_imp # Default
            if self.cat_model:
                cat_imp = dict(zip(self.feature_names, self.cat_model.get_feature_importance()))

            lgb_imp = xgb_imp # Default
            if self.lgb_model:
                lgb_imp = dict(zip(self.feature_names, self.lgb_model.feature_importances_))

            avg_importance = {}
            for feature in self.feature_names:
                tot = xgb_imp.get(feature,0) + et_imp.get(feature,0) + cat_imp.get(feature,0) + lgb_imp.get(feature,0)
                avg_importance[feature] = tot / 4.0

            return {
                'xgb_importance': xgb_imp,
                'cat_importance': cat_imp,
                'lgb_importance': lgb_imp,
                'et_importance': et_imp,
                'avg_importance': avg_importance
            }
        except Exception as e:
            if self.verbose: tprint_error(f"Error getting feature importance: {e}")
            return {}

    def ensemble_predict(self, X_non_causal: pd.DataFrame, method: str = 'performance_weighted') -> np.ndarray:
        """Ensemble predictions."""
        preds = self.predict(X_non_causal, return_confidence=False)

        if method == 'equal':
            return np.mean(list(preds.values()), axis=0)
        elif method == 'performance_weighted':
            # Simple inverse MSE weighting based on CV scores
            scores = {
                'xgb': 1.0/self.cv_scores['xgb_cv_mse'],
                'et': 1.0/self.cv_scores['et_cv_mse'],
            }
            if self.cat_model: scores['cat'] = 1.0/self.cv_scores['cat_cv_mse']
            if self.lgb_model: scores['lgb'] = 1.0/self.cv_scores['lgb_cv_mse']

            total_w = sum(scores.values())
            ensemble = np.zeros_like(preds['xgb'])
            for k, v in preds.items():
                if k in scores:
                    ensemble += v * (scores[k] / total_w)
            return ensemble
        else:
            return preds['xgb'] # Fallback

    def detect_conflict(self, *args, **kwargs):
        # Legacy method signature adapter
        return self.detect_conflict_enhanced(*args, **kwargs)

    def detect_conflict_enhanced(
        self,
        chaser_predictions: Dict[str, np.ndarray],
        causal_anchor_prediction: np.ndarray,
        chaser_confidence: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Enhanced conflict detection."""
        # Use ensemble for consensus
        # Reconstruct ensemble from predictions directly to avoid re-prediction overhead
        ensemble_pred = np.mean(list(chaser_predictions.values()), axis=0)

        conflict_results = {}
        for model_name, chaser_prediction in chaser_predictions.items():
            # Total prediction (Anchor + Chaser)
            total_prediction = causal_anchor_prediction + chaser_prediction

            # Conflict detection: Chaser betting against Anchor
            conflict_direction = np.sign(chaser_prediction) != np.sign(causal_anchor_prediction)
            conflict_magnitude = np.abs(chaser_prediction) / (np.abs(causal_anchor_prediction) + 1e-8)

            # Conflict flag (high confidence + opposite direction)
            conflict_flag = conflict_direction & (chaser_confidence > self.confidence_threshold)

            # Conflict intensity (weighted by confidence and magnitude)
            conflict_intensity = conflict_flag.astype(float) * chaser_confidence * conflict_magnitude

            # Enhanced: Disagreement with ensemble
            ensemble_disagreement = np.sign(chaser_prediction) != np.sign(ensemble_pred)
            ensemble_magnitude = np.abs(chaser_prediction - ensemble_pred) / (np.abs(ensemble_pred) + 1e-8)

            conflict_results[model_name] = {
                'conflict_flag': conflict_flag,
                'conflict_intensity': conflict_intensity,
                'conflict_direction': conflict_direction.astype(int),
                'conflict_magnitude': conflict_magnitude,
                'total_prediction': total_prediction,
                'ensemble_disagreement': ensemble_disagreement.astype(int),
                'ensemble_magnitude': ensemble_magnitude
            }

        return conflict_results

    def detect_and_handle_outliers(
        self,
        y_residuals: pd.Series,
        method: str = 'iqr',
        threshold: float = 1.5,
        handling_strategy: str = 'winsorize',
        contamination: float = 0.05
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect and handle outliers in residual data.

        Args:
            y_residuals: Target residuals
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection (IQR multiplier or z-score)
            handling_strategy: How to handle outliers ('remove', 'winsorize', 'transform')
            contamination: Expected proportion of outliers (for isolation forest)

        Returns:
            Tuple of (clean_residuals, outlier_mask)
        """
        try:
            if self.verbose:
                tprint_info(f"🔍 Detecting outliers using {method} method...")

            outlier_mask = pd.Series(False, index=y_residuals.index)

            if method == 'iqr':
                # IQR-based outlier detection
                Q1 = y_residuals.quantile(0.25)
                Q3 = y_residuals.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (y_residuals < lower_bound) | (y_residuals > upper_bound)

            elif method == 'zscore':
                # Z-score based outlier detection
                from scipy import stats
                z_scores = np.abs(stats.zscore(y_residuals))
                outlier_mask = z_scores > threshold

            elif method == 'isolation_forest':
                # Isolation Forest for outlier detection
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
                # Need to reshape for sklearn
                outlier_predictions = iso_forest.fit_predict(y_residuals.values.reshape(-1, 1))
                outlier_mask = outlier_predictions == -1

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            n_outliers = outlier_mask.sum()

            if self.verbose:
                tprint_warning(f"⚠️  Detected {n_outliers} outliers ({n_outliers/len(y_residuals)*100:.2f}%) using {method}")

            # Handle outliers based on strategy
            if handling_strategy == 'remove':
                clean_residuals = y_residuals[~outlier_mask]
                if self.verbose:
                    tprint_info(f"   🗑️  Removed {n_outliers} outliers, {len(clean_residuals)} samples remaining")

            elif handling_strategy == 'winsorize':
                # Winsorize to percentile bounds
                lower_bound = y_residuals[~outlier_mask].quantile(0.05)
                upper_bound = y_residuals[~outlier_mask].quantile(0.95)
                clean_residuals = np.clip(y_residuals, lower_bound, upper_bound)
                if self.verbose:
                    tprint_info(f"   ✂️  Winsorized outliers to [{lower_bound:.6f}, {upper_bound:.6f}] range")

            elif handling_strategy == 'transform':
                # Log transformation for positive outliers (assuming residuals can be positive/negative)
                clean_residuals = y_residuals.copy()
                # Apply log transformation to extreme values
                extreme_mask = outlier_mask & (np.abs(y_residuals) > y_residuals.std() * 2)
                if extreme_mask.any():
                    # Use signed log transformation
                    signs = np.sign(clean_residuals[extreme_mask])
                    transformed = np.log1p(np.abs(clean_residuals[extreme_mask]))
                    clean_residuals.loc[extreme_mask] = signs * transformed
                    if self.verbose:
                        tprint_info(f"   🔄 Log-transformed {extreme_mask.sum()} extreme outliers")

            else:
                raise ValueError(f"Unknown outlier handling strategy: {handling_strategy}")

            return clean_residuals, outlier_mask

        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Outlier detection/handling failed: {e}")
            raise

    def get_model_comparison(self) -> Dict[str, Any]:
        """
        Get comprehensive comparison between all three models.
        
        Returns:
            Dictionary with detailed model comparison metrics
        """
        try:
            if self.xgb_model is None:
                raise ValueError("Models not fitted yet")
            
            comparison = {
                'training_performance': self.training_score,
                'cv_performance': self.cv_scores,
                'feature_importance_correlation': 0.0, # Placeholder
                'best_model': {'name': 'Ensemble'} # Placeholder
            }
            
            return comparison
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Model comparison failed: {e}")
            raise


# Convenience functions (kept for compatibility)
def create_chaser(**kwargs):
    return Layer25Chaser(**kwargs)

def quick_chaser_fit(X_non_causal, y_residuals, **kwargs):
    chaser = create_chaser(**kwargs)
    chaser.fit(X_non_causal, y_residuals)
    return chaser
