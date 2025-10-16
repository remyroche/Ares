"""
Multi-Stage Recursive Feature Elimination (RFE)

This module implements a sophisticated multi-stage RFE approach with:
- Stage 1: mRMR pre-filtering
- Stage 2: Ensemble filtering with CV
- Stage 3: RFE in batches with CV
- Stage 4: Individual RFE with stability selection
- Automated stopping criteria with plateau detection
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

# Import LightGBM and SHAP
try:
    import lightgbm as lgb
    import shap
    LIGHTGBM_AVAILABLE = True
    SHAP_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    SHAP_AVAILABLE = False

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

# Import pre-filtering
from .prefiltering import MRMRSpearmanPreFilter

logger = logging.getLogger(__name__)

class MultiStageRFE:
    """Multi-stage RFE with comprehensive validation and stability selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-stage RFE."""
        self.config = config or {
            'target_features': None,
            'enable_prefiltering': True,
            'enable_plateau_detection': True,
            'plateau_threshold': 0.01,  # 1% improvement threshold
            'plateau_patience': 3,  # Number of iterations to wait
            'cv_folds': 5,
            'cv_strategy': 'kfold',  # 'kfold', 'stratified', 'timeseries'
            'stability_threshold': 0.8,
            'stability_n_bootstrap': 10,
            'enable_hardware_optimization': True,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': True
        }

        self.logger = logger.getChild('MultiStageRFE')

        # Initialize hardware optimization
        if self.config.get('enable_hardware_optimization', True):
            self.cpu_optimizer = M1CPUOptimizer()
            hw_config = HardwareConfig(
                cpu_optimization_level='aggressive',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None

        # Initialize pre-filter
        if self.config.get('enable_prefiltering', True):
            self.pre_filter = MRMRSpearmanPreFilter({
                'mrmr_weight': 0.7,
                'spearman_weight': 0.3,
                'enable_hardware_optimization': self.config.get('enable_hardware_optimization', True),
                'n_jobs': self.config.get('n_jobs', -1),
                'random_state': self.config.get('random_state', 42)
            })
        else:
            self.pre_filter = None

        # Performance tracking
        self.performance_stats = {
            'total_runs': 0,
            'stage1_completions': 0,
            'stage2_completions': 0,
            'stage3_completions': 0,
            'stage4_completions': 0,
            'plateau_detections': 0,
            'avg_total_time': 0.0
        }

        # Plateau detection
        self.plateau_detector = PlateauDetector(
            threshold=self.config.get('plateau_threshold', 0.01),
            patience=self.config.get('plateau_patience', 3)
        )

        tprint_success("🔧 MultiStageRFE initialized")

    def select_features(self, X: np.ndarray, y: np.ndarray,
                       target_features: int,
                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Select features using multi-stage RFE approach."""
        tprint_info(f"🔧 Multi-stage RFE selection: {X.shape} -> {target_features} features")

        start_time = time.time()

        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Update target features
            self.config['target_features'] = target_features

            # Stage 1: mRMR pre-filtering
            stage1_result = self._stage1_prefiltering(X, y, target_features, feature_names)
            if not stage1_result['success']:
                return stage1_result

            X_stage1 = stage1_result['X_filtered']
            feature_names_stage1 = stage1_result['filtered_feature_names']
            stage1_mask = stage1_result['feature_mask']

            # Stage 2: Ensemble filtering
            stage2_result = self._stage2_ensemble_filtering(
                X_stage1, y, target_features, feature_names_stage1
            )
            if not stage2_result['success']:
                return stage2_result

            X_stage2 = stage2_result['X_filtered']
            feature_names_stage2 = stage2_result['filtered_feature_names']
            stage2_mask = stage2_result['feature_mask']

            # Stage 3: RFE in batches
            stage3_result = self._stage3_rfe_batches(
                X_stage2, y, target_features, feature_names_stage2
            )
            if not stage3_result['success']:
                return stage3_result

            X_stage3 = stage3_result['X_filtered']
            feature_names_stage3 = stage3_result['filtered_feature_names']
            stage3_mask = stage3_result['feature_mask']

            # Stage 4: Individual RFE
            stage4_result = self._stage4_individual_rfe(
                X_stage3, y, target_features, feature_names_stage3
            )
            if not stage4_result['success']:
                return stage4_result

            # Combine all stage masks
            final_mask = self._combine_stage_masks(
                stage1_mask, stage2_mask, stage3_mask, stage4_result['feature_mask']
            )

            # Get final selected features
            selected_indices = np.where(final_mask)[0]
            selected_features = [feature_names[i] for i in selected_indices]

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_runs'] += 1
            self.performance_stats['avg_total_time'] = (
                (self.performance_stats['avg_total_time'] * (self.performance_stats['total_runs'] - 1) +
                 execution_time) / self.performance_stats['total_runs']
            )

            result = {
                'success': True,
                'selected_features': selected_features,
                'selected_indices': selected_indices.tolist(),
                'n_selected': len(selected_features),
                'n_total': X.shape[1],
                'selection_ratio': len(selected_features) / X.shape[1],
                'stage_results': {
                    'stage1': stage1_result,
                    'stage2': stage2_result,
                    'stage3': stage3_result,
                    'stage4': stage4_result
                },
                'execution_time': execution_time,
                'method': 'multi_stage_rfe'
            }

            tprint_success(f"✅ Multi-stage RFE completed: {X.shape[1]} -> {len(selected_features)} features in {execution_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"Multi-stage RFE failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _stage1_prefiltering(self, X: np.ndarray, y: np.ndarray,
                           target_features: int, feature_names: List[str]) -> Dict[str, Any]:
        """Stage 1: mRMR pre-filtering to top 50% of (total - target) features."""
        tprint_info("🔧 Stage 1: mRMR pre-filtering")

        if not self.pre_filter:
            # Skip pre-filtering
            return {
                'success': True,
                'X_filtered': X,
                'feature_mask': np.ones(X.shape[1], dtype=bool),
                'filtered_feature_names': feature_names,
                'stage': 'stage1_skipped'
            }

        try:
            result = self.pre_filter.prefilter_features(X, y, target_features, feature_names)
            self.performance_stats['stage1_completions'] += 1
            return result

        except Exception as e:
            self.logger.error(f"Stage 1 failed: {e}")
            return {'success': False, 'error': str(e)}

    def _stage2_ensemble_filtering(self, X: np.ndarray, y: np.ndarray,
                                 target_features: int, feature_names: List[str]) -> Dict[str, Any]:
        """Stage 2: Ensemble filtering to top 25% with CV until total + 60."""
        tprint_info("🔧 Stage 2: Ensemble filtering")

        try:
            n_current = X.shape[1]
            n_target = target_features
            n_buffer = 60  # Keep 60 more than target

            # Calculate target for this stage
            target_this_stage = min(n_current, n_target + n_buffer)

            # If we already have fewer features than target, return as is
            if n_current <= target_this_stage:
                return {
                    'success': True,
                    'X_filtered': X,
                    'feature_mask': np.ones(n_current, dtype=bool),
                    'filtered_feature_names': feature_names,
                    'stage': 'stage2_skipped'
                }

            # Use ensemble approach (LGBM SHAP + LASSO + RandomForest)
            ensemble_scores = self._calculate_ensemble_scores(X, y, feature_names)

            # Select top features
            sorted_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
            n_select = min(target_this_stage, len(sorted_features))

            selected_features = [f[0] for f in sorted_features[:n_select]]
            selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]

            # Create feature mask
            feature_mask = np.zeros(n_current, dtype=bool)
            feature_mask[selected_indices] = True

            # Get filtered data
            X_filtered = X[:, selected_indices]
            filtered_feature_names = [feature_names[i] for i in selected_indices]

            self.performance_stats['stage2_completions'] += 1

            return {
                'success': True,
                'X_filtered': X_filtered,
                'feature_mask': feature_mask,
                'filtered_feature_names': filtered_feature_names,
                'ensemble_scores': ensemble_scores,
                'stage': 'stage2_ensemble'
            }

        except Exception as e:
            self.logger.error(f"Stage 2 failed: {e}")
            return {'success': False, 'error': str(e)}

    def _stage3_rfe_batches(self, X: np.ndarray, y: np.ndarray,
                          target_features: int, feature_names: List[str]) -> Dict[str, Any]:
        """Stage 3: RFE in 10% batches with CV until total + 20."""
        tprint_info("🔧 Stage 3: RFE in batches")

        try:
            n_current = X.shape[1]
            n_target = target_features
            n_buffer = 20  # Keep 20 more than target

            # Calculate target for this stage
            target_this_stage = min(n_current, n_target + n_buffer)

            # If we already have fewer features than target, return as is
            if n_current <= target_this_stage:
                return {
                    'success': True,
                    'X_filtered': X,
                    'feature_mask': np.ones(n_current, dtype=bool),
                    'filtered_feature_names': feature_names,
                    'stage': 'stage3_skipped'
                }

            # Use RFE with batch removal
            X_current = X.copy()
            feature_names_current = feature_names.copy()
            current_indices = list(range(n_current))

            while X_current.shape[1] > target_this_stage:
                # Calculate batch size (10% of remaining features)
                batch_size = max(1, int(0.1 * X_current.shape[1]))
                batch_size = min(batch_size, X_current.shape[1] - target_this_stage)

                # Use RFE to remove batch_size features
                rfe_result = self._rfe_remove_batch(
                    X_current, y, batch_size, feature_names_current
                )

                if not rfe_result['success']:
                    break

                # Update current data
                X_current = rfe_result['X_filtered']
                feature_names_current = rfe_result['filtered_feature_names']
                current_indices = rfe_result['selected_indices']

                # Check plateau detection
                if self.config.get('enable_plateau_detection', True):
                    if self.plateau_detector.check_plateau(rfe_result.get('cv_score', 0.0)):
                        tprint_info("🔧 Plateau detected, stopping Stage 3")
                        self.performance_stats['plateau_detections'] += 1
                        break

            # Create feature mask
            feature_mask = np.zeros(n_current, dtype=bool)
            feature_mask[current_indices] = True

            self.performance_stats['stage3_completions'] += 1

            return {
                'success': True,
                'X_filtered': X_current,
                'feature_mask': feature_mask,
                'filtered_feature_names': feature_names_current,
                'stage': 'stage3_rfe_batches'
            }

        except Exception as e:
            self.logger.error(f"Stage 3 failed: {e}")
            return {'success': False, 'error': str(e)}

    def _stage4_individual_rfe(self, X: np.ndarray, y: np.ndarray,
                             target_features: int, feature_names: List[str]) -> Dict[str, Any]:
        """Stage 4: Individual RFE with stability selection."""
        tprint_info("🔧 Stage 4: Individual RFE")

        try:
            n_current = X.shape[1]
            n_target = target_features

            # If we already have target features or fewer, return as is
            if n_current <= n_target:
                return {
                    'success': True,
                    'X_filtered': X,
                    'feature_mask': np.ones(n_current, dtype=bool),
                    'filtered_feature_names': feature_names,
                    'stage': 'stage4_skipped'
                }

            # Use individual RFE with stability selection
            rfe_result = self._rfe_individual_with_stability(
                X, y, n_target, feature_names
            )

            if not rfe_result['success']:
                return rfe_result

            self.performance_stats['stage4_completions'] += 1

            return {
                'success': True,
                'X_filtered': rfe_result['X_filtered'],
                'feature_mask': rfe_result['feature_mask'],
                'filtered_feature_names': rfe_result['filtered_feature_names'],
                'stability_scores': rfe_result.get('stability_scores', {}),
                'stage': 'stage4_individual_rfe'
            }

        except Exception as e:
            self.logger.error(f"Stage 4 failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_ensemble_scores(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, float]:
        """Calculate ensemble scores using LGBM SHAP + LASSO + RandomForest."""
        tprint_debug("🔧 Calculating ensemble scores")

        try:
            ensemble_scores = {}
            n_features = X.shape[1]

            # Initialize scores
            for feature_name in feature_names:
                ensemble_scores[feature_name] = 0.0

            # LGBM SHAP scores
            if LIGHTGBM_AVAILABLE and SHAP_AVAILABLE:
                lgb_scores = self._calculate_lgb_shap_scores(X, y, feature_names)
                for feature_name, score in lgb_scores.items():
                    ensemble_scores[feature_name] += 0.4 * score  # 40% weight

            # LASSO scores
            lasso_scores = self._calculate_lasso_scores(X, y, feature_names)
            for feature_name, score in lasso_scores.items():
                ensemble_scores[feature_name] += 0.3 * score  # 30% weight

            # RandomForest scores
            rf_scores = self._calculate_rf_scores(X, y, feature_names)
            for feature_name, score in rf_scores.items():
                ensemble_scores[feature_name] += 0.3 * score  # 30% weight

            return ensemble_scores

        except Exception as e:
            self.logger.warning(f"Ensemble score calculation failed: {e}")
            # Fallback to simple RandomForest scores
            return self._calculate_rf_scores(X, y, feature_names)

    def _calculate_lgb_shap_scores(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str]) -> Dict[str, float]:
        """Calculate LGBM SHAP-based importance scores."""
        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            if is_classification:
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    verbose=-1
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    verbose=-1
                )

            # Fit model
            model.fit(X, y)

            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # Calculate mean absolute SHAP values
            if is_classification and len(shap_values) > 1:
                # Multi-class classification
                shap_values = np.array(shap_values)
                mean_shap = np.mean(np.abs(shap_values), axis=0)
            else:
                # Binary classification or regression
                mean_shap = np.mean(np.abs(shap_values), axis=0)

            # Create scores dictionary
            shap_scores = {}
            for i, feature_name in enumerate(feature_names):
                shap_scores[feature_name] = float(mean_shap[i])

            return shap_scores

        except Exception as e:
            self.logger.warning(f"LGBM SHAP calculation failed: {e}")
            # Fallback to LGBM feature importance
            return self._calculate_lgb_importance_scores(X, y, feature_names)

    def _calculate_lgb_importance_scores(self, X: np.ndarray, y: np.ndarray,
                                       feature_names: List[str]) -> Dict[str, float]:
        """Calculate LGBM feature importance scores as fallback."""
        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            if is_classification:
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    verbose=-1
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    verbose=-1
                )

            # Fit model
            model.fit(X, y)

            # Get feature importance
            importance = model.feature_importances_

            # Create scores dictionary
            importance_scores = {}
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = float(importance[i])

            return importance_scores

        except Exception as e:
            self.logger.warning(f"LGBM importance calculation failed: {e}")
            return {feature_name: 0.0 for feature_name in feature_names}

    def _calculate_lasso_scores(self, X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> Dict[str, float]:
        """Calculate LASSO-based importance scores."""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit LASSO with cross-validation
            lasso = LassoCV(
                cv=self.config.get('cv_folds', 5),
                random_state=self.config.get('random_state', 42),
                n_jobs=self.config.get('n_jobs', -1)
            )
            lasso.fit(X_scaled, y)

            # Get coefficients as importance
            coefficients = np.abs(lasso.coef_)

            # Create scores dictionary
            lasso_scores = {}
            for i, feature_name in enumerate(feature_names):
                lasso_scores[feature_name] = float(coefficients[i])

            return lasso_scores

        except Exception as e:
            self.logger.warning(f"LASSO calculation failed: {e}")
            return {feature_name: 0.0 for feature_name in feature_names}

    def _calculate_rf_scores(self, X: np.ndarray, y: np.ndarray,
                           feature_names: List[str]) -> Dict[str, float]:
        """Calculate RandomForest-based importance scores."""
        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1)
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1)
                )

            # Fit model
            model.fit(X, y)

            # Get feature importance
            importance = model.feature_importances_

            # Create scores dictionary
            rf_scores = {}
            for i, feature_name in enumerate(feature_names):
                rf_scores[feature_name] = float(importance[i])

            return rf_scores

        except Exception as e:
            self.logger.warning(f"RandomForest calculation failed: {e}")
            return {feature_name: 0.0 for feature_name in feature_names}

    def _rfe_remove_batch(self, X: np.ndarray, y: np.ndarray,
                         batch_size: int, feature_names: List[str]) -> Dict[str, Any]:
        """Remove a batch of features using RFE."""
        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            if is_classification:
                estimator = RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.config.get('random_state', 42)
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.config.get('random_state', 42)
                )

            # Use RFE to remove batch_size features
            n_features_to_select = X.shape[1] - batch_size
            rfe = RFE(estimator, n_features_to_select=n_features_to_select)
            rfe.fit(X, y)

            # Get selected features
            selected_mask = rfe.support_
            selected_indices = np.where(selected_mask)[0]

            # Get filtered data
            X_filtered = X[:, selected_indices]
            filtered_feature_names = [feature_names[i] for i in selected_indices]

            # Calculate CV score for plateau detection
            cv_score = self._calculate_cv_score(X_filtered, y, is_classification)

            return {
                'success': True,
                'X_filtered': X_filtered,
                'selected_indices': selected_indices.tolist(),
                'filtered_feature_names': filtered_feature_names,
                'cv_score': cv_score
            }

        except Exception as e:
            self.logger.warning(f"RFE batch removal failed: {e}")
            return {'success': False, 'error': str(e)}

    def _rfe_individual_with_stability(self, X: np.ndarray, y: np.ndarray,
                                     target_features: int, feature_names: List[str]) -> Dict[str, Any]:
        """Individual RFE with stability selection."""
        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            if is_classification:
                estimator = RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.config.get('random_state', 42)
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50,
                    random_state=self.config.get('random_state', 42)
                )

            # Use RFECV for individual feature removal with CV
            rfecv = RFECV(
                estimator,
                step=1,
                cv=self.config.get('cv_folds', 5),
                scoring='accuracy' if is_classification else 'r2',
                n_jobs=self.config.get('n_jobs', -1)
            )
            rfecv.fit(X, y)

            # Get selected features
            selected_mask = rfecv.support_
            selected_indices = np.where(selected_mask)[0]

            # Get filtered data
            X_filtered = X[:, selected_indices]
            filtered_feature_names = [feature_names[i] for i in selected_indices]

            # Calculate stability scores
            stability_scores = self._calculate_stability_scores(X, y, selected_indices, feature_names)

            return {
                'success': True,
                'X_filtered': X_filtered,
                'feature_mask': selected_mask,
                'selected_indices': selected_indices.tolist(),
                'filtered_feature_names': filtered_feature_names,
                'stability_scores': stability_scores,
                'cv_scores': rfecv.cv_results_
            }

        except Exception as e:
            self.logger.error(f"Individual RFE with stability failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_stability_scores(self, X: np.ndarray, y: np.ndarray,
                                  selected_indices: np.ndarray,
                                  feature_names: List[str]) -> Dict[str, float]:
        """Calculate stability scores for selected features."""
        try:
            n_bootstrap = self.config.get('stability_n_bootstrap', 10)
            stability_scores = {}

            for i, feature_idx in enumerate(selected_indices):
                feature_name = feature_names[feature_idx]
                stability_count = 0

                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    n_samples = X.shape[0]
                    bootstrap_indices = np.random.choice(
                        n_samples, size=n_samples, replace=True
                    )
                    X_bootstrap = X[bootstrap_indices]
                    y_bootstrap = y[bootstrap_indices]

                    # Calculate feature importance for bootstrap
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    rf.fit(X_bootstrap, y_bootstrap)
                    importance = rf.feature_importances_

                    # Check if this feature is in top features
                    top_features = np.argsort(importance)[-len(selected_indices):]
                    if feature_idx in top_features:
                        stability_count += 1

                stability_scores[feature_name] = stability_count / n_bootstrap

            return stability_scores

        except Exception as e:
            self.logger.warning(f"Stability score calculation failed: {e}")
            return {feature_names[i]: 0.0 for i in selected_indices}

    def _calculate_cv_score(self, X: np.ndarray, y: np.ndarray,
                          is_classification: bool) -> float:
        """Calculate cross-validation score for plateau detection."""
        try:
            if is_classification:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                scoring = 'accuracy'
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                scoring = 'r2'

            scores = cross_val_score(
                estimator, X, y,
                cv=self.config.get('cv_folds', 5),
                scoring=scoring,
                n_jobs=1  # Avoid nested parallelism
            )

            return float(np.mean(scores))

        except Exception as e:
            self.logger.warning(f"CV score calculation failed: {e}")
            return 0.0

    def _combine_stage_masks(self, stage1_mask: np.ndarray, stage2_mask: np.ndarray,
                           stage3_mask: np.ndarray, stage4_mask: np.ndarray) -> np.ndarray:
        """Combine masks from all stages."""
        try:
            # Start with stage1 mask
            final_mask = stage1_mask.copy()

            # Apply stage2 mask (if it's a subset of stage1)
            if len(stage2_mask) <= len(final_mask):
                stage2_indices = np.where(stage2_mask)[0]
                final_mask = np.zeros_like(final_mask)
                final_mask[stage2_indices] = True

            # Apply stage3 mask (if it's a subset of stage2)
            if len(stage3_mask) <= len(final_mask):
                stage3_indices = np.where(stage3_mask)[0]
                final_mask = np.zeros_like(final_mask)
                final_mask[stage3_indices] = True

            # Apply stage4 mask (if it's a subset of stage3)
            if len(stage4_mask) <= len(final_mask):
                stage4_indices = np.where(stage4_mask)[0]
                final_mask = np.zeros_like(final_mask)
                final_mask[stage4_indices] = True

            return final_mask

        except Exception as e:
            self.logger.warning(f"Mask combination failed: {e}")
            return stage4_mask  # Return the most refined mask

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_runs'] > 0:
            stats['stage1_success_rate'] = stats['stage1_completions'] / stats['total_runs']
            stats['stage2_success_rate'] = stats['stage2_completions'] / stats['total_runs']
            stats['stage3_success_rate'] = stats['stage3_completions'] / stats['total_runs']
            stats['stage4_success_rate'] = stats['stage4_completions'] / stats['total_runs']
            stats['plateau_detection_rate'] = stats['plateau_detections'] / stats['total_runs']
        else:
            stats['stage1_success_rate'] = 0.0
            stats['stage2_success_rate'] = 0.0
            stats['stage3_success_rate'] = 0.0
            stats['stage4_success_rate'] = 0.0
            stats['plateau_detection_rate'] = 0.0

        return stats

class PlateauDetector:
    """Detect performance plateaus for automated stopping."""

    def __init__(self, threshold: float = 0.01, patience: int = 3):
        """Initialize plateau detector."""
        self.threshold = threshold
        self.patience = patience
        self.scores_history = []
        self.plateau_count = 0

    def check_plateau(self, current_score: float) -> bool:
        """Check if performance has plateaued."""
        self.scores_history.append(current_score)

        if len(self.scores_history) < 2:
            return False

        # Check if improvement is below threshold
        recent_improvement = self.scores_history[-1] - self.scores_history[-2]

        if recent_improvement < self.threshold:
            self.plateau_count += 1
        else:
            self.plateau_count = 0

        # Return True if plateau detected
        return self.plateau_count >= self.patience

    def reset(self):
        """Reset plateau detector."""
        self.scores_history = []
        self.plateau_count = 0

def create_multi_stage_rfe(config: Optional[Dict[str, Any]] = None) -> MultiStageRFE:
    """Create a multi-stage RFE selector."""
    return MultiStageRFE(config)
