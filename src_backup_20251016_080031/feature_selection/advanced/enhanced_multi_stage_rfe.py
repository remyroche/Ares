"""
Enhanced Multi-Stage RFE with SHAP, Z-score Normalization, and Stability Selection

This module implements the enhanced multi-stage RFE approach with:
- Stage 1: mRMR pre-filtering to top 50%
- Stage 2: Ensemble filtering with LGBM SHAP + LASSO + RandomForest
- Stage 3: Batch RFE with CV and stability selection
- Stage 4: Fine RFE one-by-one with plateau detection
- Z-score normalization and stability frequency tracking
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, GroupKFold
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from scipy.stats import rankdata, zscore

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

# Import improved mRMR
from .improved_mrmr import ImprovedMRMR

logger = logging.getLogger(__name__)

class EnhancedMultiStageRFE:
    """Enhanced multi-stage RFE with SHAP, z-score normalization, and stability selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced multi-stage RFE."""
        self.config = config or {
            'target_features': None,
            'enable_stage1': True,
            'enable_stage2': True,
            'enable_stage3': True,
            'enable_stage4': True,
            'stage2_buffer': 60,  # Keep 60 more than target
            'stage3_buffer': 20,  # Keep 20 more than target
            'stage2_ratio': 0.25,  # Keep top 25% in each iteration
            'stage3_batch_ratio': 0.1,  # Remove 10% in each batch
            'stability_threshold': 0.6,  # Minimum stability frequency
            'high_stability_threshold': 0.9,  # Lock high stability features
            'plateau_threshold': 0.002,  # AUC improvement threshold
            'plateau_patience': 2,  # Patience for plateau detection
            'cv_folds': 5,
            'cv_strategy': 'stratified',  # 'stratified', 'kfold', 'grouped', 'timeseries'
            'enable_bootstrap': True,
            'bootstrap_samples': 3,
            'lgb_params': {
                'max_depth': 8,
                'num_leaves': 256,  # 2^8
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': 42,
                'verbose': -1
            },
            'enable_hardware_optimization': True,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': True
        }
        
        self.logger = logger.getChild('EnhancedMultiStageRFE')
        
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
        
        # Initialize mRMR pre-filter
        if self.config.get('enable_stage1', True):
            self.mrmr_filter = ImprovedMRMR({
                'target_ratio': 0.5,  # Select top 50%
                'mi_weight': 0.7,
                'spearman_weight': 0.3,
                'enable_hardware_optimization': self.config.get('enable_hardware_optimization', True),
                'n_jobs': self.config.get('n_jobs', -1),
                'random_state': self.config.get('random_state', 42)
            })
        else:
            self.mrmr_filter = None
        
        # Plateau detector
        self.plateau_detector = PlateauDetector(
            threshold=self.config.get('plateau_threshold', 0.002),
            patience=self.config.get('plateau_patience', 2)
        )
        
        # Performance tracking
        self.performance_stats = {
            'total_runs': 0,
            'stage1_completions': 0,
            'stage2_completions': 0,
            'stage3_completions': 0,
            'stage4_completions': 0,
            'plateau_detections': 0,
            'stability_locks': 0,
            'avg_total_time': 0.0
        }
        
        tprint_success("🔧 EnhancedMultiStageRFE initialized")
    
    def select_features(self, X: np.ndarray, y: np.ndarray,
                       target_features: int,
                       feature_names: Optional[List[str]] = None,
                       groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Select features using enhanced multi-stage RFE approach."""
        tprint_info(f"🔧 Enhanced multi-stage RFE: {X.shape} -> {target_features} features")
        
        start_time = time.time()
        
        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Update target features
            self.config['target_features'] = target_features
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))
            
            # Stage 1: mRMR pre-filtering
            stage1_result = self._stage1_mrmr_prefiltering(X, y, feature_names)
            if not stage1_result['success']:
                return stage1_result
            
            X_stage1 = stage1_result['X_filtered']
            feature_names_stage1 = stage1_result['filtered_feature_names']
            stage1_mask = stage1_result['feature_mask']
            
            # Stage 2: Ensemble filtering
            stage2_result = self._stage2_ensemble_filtering(
                X_stage1, y, target_features, feature_names_stage1, 
                is_classification, groups
            )
            if not stage2_result['success']:
                return stage2_result
            
            X_stage2 = stage2_result['X_filtered']
            feature_names_stage2 = stage2_result['filtered_feature_names']
            stage2_mask = stage2_result['feature_mask']
            
            # Stage 3: Batch RFE
            stage3_result = self._stage3_batch_rfe(
                X_stage2, y, target_features, feature_names_stage2,
                is_classification, groups
            )
            if not stage3_result['success']:
                return stage3_result
            
            X_stage3 = stage3_result['X_filtered']
            feature_names_stage3 = stage3_result['filtered_feature_names']
            stage3_mask = stage3_result['feature_mask']
            
            # Stage 4: Fine RFE
            stage4_result = self._stage4_fine_rfe(
                X_stage3, y, target_features, feature_names_stage3,
                is_classification, groups
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
                'method': 'enhanced_multi_stage_rfe'
            }
            
            tprint_success(f"✅ Enhanced multi-stage RFE completed: {X.shape[1]} -> {len(selected_features)} features in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced multi-stage RFE failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _stage1_mrmr_prefiltering(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str]) -> Dict[str, Any]:
        """Stage 1: mRMR pre-filtering to top 50%."""
        tprint_info("🔧 Stage 1: mRMR pre-filtering")
        
        if not self.mrmr_filter:
            return {
                'success': True,
                'X_filtered': X,
                'feature_mask': np.ones(X.shape[1], dtype=bool),
                'filtered_feature_names': feature_names,
                'stage': 'stage1_skipped'
            }
        
        try:
            result = self.mrmr_filter.select_features(X, y, feature_names, target_ratio=0.5)
            self.performance_stats['stage1_completions'] += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Stage 1 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _stage2_ensemble_filtering(self, X: np.ndarray, y: np.ndarray,
                                 target_features: int, feature_names: List[str],
                                 is_classification: bool, groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Stage 2: Ensemble filtering with LGBM SHAP + LASSO + RandomForest."""
        tprint_info("🔧 Stage 2: Ensemble filtering")
        
        try:
            n_current = X.shape[1]
            n_target = target_features
            n_buffer = self.config.get('stage2_buffer', 60)
            target_this_stage = min(n_current, n_target + n_buffer)
            
            if n_current <= target_this_stage:
                return {
                    'success': True,
                    'X_filtered': X,
                    'feature_mask': np.ones(n_current, dtype=bool),
                    'filtered_feature_names': feature_names,
                    'stage': 'stage2_skipped'
                }
            
            # Iterate until we reach target
            X_current = X.copy()
            feature_names_current = feature_names.copy()
            current_indices = list(range(n_current))
            
            iteration = 0
            while X_current.shape[1] > target_this_stage:
                iteration += 1
                tprint_debug(f"🔧 Stage 2 iteration {iteration}: {X_current.shape[1]} features")
                
                # Calculate ensemble scores with CV
                ensemble_scores = self._calculate_ensemble_scores_cv(
                    X_current, y, feature_names_current, is_classification, groups
                )
                
                # Select top 25% of current features
                n_select = max(target_this_stage, int(self.config.get('stage2_ratio', 0.25) * X_current.shape[1]))
                n_select = min(n_select, X_current.shape[1])
                
                # Sort features by ensemble scores
                sorted_features = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:n_select]]
                
                # Update current data
                selected_indices = [feature_names_current.index(f) for f in selected_features if f in feature_names_current]
                X_current = X_current[:, selected_indices]
                feature_names_current = [feature_names_current[i] for i in selected_indices]
                current_indices = [current_indices[i] for i in selected_indices]
                
                # Check plateau detection
                if self.config.get('enable_plateau_detection', True):
                    cv_score = self._calculate_cv_score(X_current, y, is_classification, groups)
                    if self.plateau_detector.check_plateau(cv_score):
                        tprint_info("🔧 Plateau detected in Stage 2, stopping early")
                        self.performance_stats['plateau_detections'] += 1
                        break
            
            # Create feature mask
            feature_mask = np.zeros(n_current, dtype=bool)
            feature_mask[current_indices] = True
            
            self.performance_stats['stage2_completions'] += 1
            
            return {
                'success': True,
                'X_filtered': X_current,
                'feature_mask': feature_mask,
                'filtered_feature_names': feature_names_current,
                'stage': 'stage2_ensemble',
                'iterations': iteration
            }
            
        except Exception as e:
            self.logger.error(f"Stage 2 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _stage3_batch_rfe(self, X: np.ndarray, y: np.ndarray,
                        target_features: int, feature_names: List[str],
                        is_classification: bool, groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Stage 3: Batch RFE with CV and stability selection."""
        tprint_info("🔧 Stage 3: Batch RFE")
        
        try:
            n_current = X.shape[1]
            n_target = target_features
            n_buffer = self.config.get('stage3_buffer', 20)
            target_this_stage = min(n_current, n_target + n_buffer)
            
            if n_current <= target_this_stage:
                return {
                    'success': True,
                    'X_filtered': X,
                    'feature_mask': np.ones(n_current, dtype=bool),
                    'filtered_feature_names': feature_names,
                    'stage': 'stage3_skipped'
                }
            
            # Initialize stability tracking
            stability_counts = {name: 0 for name in feature_names}
            n_bootstrap = self.config.get('bootstrap_samples', 3)
            
            X_current = X.copy()
            feature_names_current = feature_names.copy()
            current_indices = list(range(n_current))
            
            iteration = 0
            while X_current.shape[1] > target_this_stage:
                iteration += 1
                tprint_debug(f"🔧 Stage 3 iteration {iteration}: {X_current.shape[1]} features")
                
                # Calculate batch size (10% of remaining features)
                batch_size = max(1, int(self.config.get('stage3_batch_ratio', 0.1) * X_current.shape[1]))
                batch_size = min(batch_size, X_current.shape[1] - target_this_stage)
                
                # Calculate ensemble scores with stability
                ensemble_scores = self._calculate_ensemble_scores_with_stability(
                    X_current, y, feature_names_current, is_classification, groups, n_bootstrap
                )
                
                # Update stability counts
                for feature_name, count in ensemble_scores.get('stability_counts', {}).items():
                    if feature_name in stability_counts:
                        stability_counts[feature_name] += count
                
                # Lock high stability features
                high_stability_threshold = self.config.get('high_stability_threshold', 0.9)
                locked_features = set()
                for feature_name, count in stability_counts.items():
                    if count >= high_stability_threshold * iteration * n_bootstrap:
                        locked_features.add(feature_name)
                        self.performance_stats['stability_locks'] += 1
                
                # Sort features by ensemble scores, excluding locked features
                sorted_features = sorted(ensemble_scores['scores'].items(), key=lambda x: x[1], reverse=True)
                
                # Remove bottom features (excluding locked ones)
                features_to_remove = []
                for feature_name, score in reversed(sorted_features):
                    if feature_name not in locked_features and len(features_to_remove) < batch_size:
                        features_to_remove.append(feature_name)
                
                # Update current data
                remaining_features = [f for f in feature_names_current if f not in features_to_remove]
                selected_indices = [feature_names_current.index(f) for f in remaining_features]
                
                X_current = X_current[:, selected_indices]
                feature_names_current = [feature_names_current[i] for i in selected_indices]
                current_indices = [current_indices[i] for i in selected_indices]
                
                # Check plateau detection
                if self.config.get('enable_plateau_detection', True):
                    cv_score = self._calculate_cv_score(X_current, y, is_classification, groups)
                    if self.plateau_detector.check_plateau(cv_score):
                        tprint_info("🔧 Plateau detected in Stage 3, stopping early")
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
                'stability_counts': stability_counts,
                'stage': 'stage3_batch_rfe',
                'iterations': iteration
            }
            
        except Exception as e:
            self.logger.error(f"Stage 3 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _stage4_fine_rfe(self, X: np.ndarray, y: np.ndarray,
                       target_features: int, feature_names: List[str],
                       is_classification: bool, groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Stage 4: Fine RFE one-by-one with plateau detection."""
        tprint_info("🔧 Stage 4: Fine RFE")
        
        try:
            n_current = X.shape[1]
            n_target = target_features
            
            if n_current <= n_target:
                return {
                    'success': True,
                    'X_filtered': X,
                    'feature_mask': np.ones(n_current, dtype=bool),
                    'filtered_feature_names': feature_names,
                    'stage': 'stage4_skipped'
                }
            
            # Initialize stability tracking
            stability_counts = {name: 0 for name in feature_names}
            n_bootstrap = self.config.get('bootstrap_samples', 3)
            
            X_current = X.copy()
            feature_names_current = feature_names.copy()
            current_indices = list(range(n_current))
            
            # Calculate initial CV score
            best_cv_score = self._calculate_cv_score(X_current, y, is_classification, groups)
            tolerance = self.config.get('plateau_threshold', 0.002)
            
            iteration = 0
            no_improvement = 0
            max_no_improvement = self.config.get('plateau_patience', 2)
            
            while X_current.shape[1] > n_target and no_improvement < max_no_improvement:
                iteration += 1
                tprint_debug(f"🔧 Stage 4 iteration {iteration}: {X_current.shape[1]} features")
                
                # Calculate ensemble scores with stability
                ensemble_scores = self._calculate_ensemble_scores_with_stability(
                    X_current, y, feature_names_current, is_classification, groups, n_bootstrap
                )
                
                # Update stability counts
                for feature_name, count in ensemble_scores.get('stability_counts', {}).items():
                    if feature_name in stability_counts:
                        stability_counts[feature_name] += count
                
                # Find feature with lowest score
                sorted_features = sorted(ensemble_scores['scores'].items(), key=lambda x: x[1])
                if not sorted_features:
                    break
                
                feature_to_remove = sorted_features[0][0]
                
                # Check if we can remove this feature
                if feature_to_remove not in feature_names_current:
                    break
                
                # Create candidate dataset without this feature
                candidate_indices = [i for i, name in enumerate(feature_names_current) if name != feature_to_remove]
                X_candidate = X_current[:, candidate_indices]
                
                # Calculate CV score for candidate
                candidate_cv_score = self._calculate_cv_score(X_candidate, y, is_classification, groups)
                
                # Check if removal is acceptable
                if is_classification:
                    # For classification, higher is better
                    improvement = candidate_cv_score - best_cv_score
                    acceptable = improvement >= -tolerance
                else:
                    # For regression, lower is better
                    improvement = best_cv_score - candidate_cv_score
                    acceptable = improvement >= -tolerance
                
                if acceptable:
                    # Accept removal
                    X_current = X_candidate
                    feature_names_current = [feature_names_current[i] for i in candidate_indices]
                    current_indices = [current_indices[i] for i in candidate_indices]
                    best_cv_score = candidate_cv_score
                    no_improvement = 0
                    tprint_debug(f"🔧 Removed {feature_to_remove}, new CV score: {candidate_cv_score:.4f}")
                else:
                    # Reject removal
                    no_improvement += 1
                    tprint_debug(f"🔧 Rejected removal of {feature_to_remove}, no improvement: {no_improvement}")
            
            # Create feature mask
            feature_mask = np.zeros(n_current, dtype=bool)
            feature_mask[current_indices] = True
            
            self.performance_stats['stage4_completions'] += 1
            
            return {
                'success': True,
                'X_filtered': X_current,
                'feature_mask': feature_mask,
                'filtered_feature_names': feature_names_current,
                'stability_counts': stability_counts,
                'final_cv_score': best_cv_score,
                'stage': 'stage4_fine_rfe',
                'iterations': iteration
            }
            
        except Exception as e:
            self.logger.error(f"Stage 4 failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_ensemble_scores_cv(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str], is_classification: bool,
                                    groups: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate ensemble scores with cross-validation."""
        try:
            # Create CV splits
            cv_splits = self._create_cv_splits(X, y, groups)
            
            # Calculate scores for each fold
            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Calculate ensemble scores for this fold
                fold_ensemble = self._calculate_fold_ensemble_scores(
                    X_train, y_train, X_val, y_val, feature_names, is_classification
                )
                fold_scores.append(fold_ensemble)
            
            # Average scores across folds
            ensemble_scores = {}
            for feature_name in feature_names:
                scores = [fold.get(feature_name, 0.0) for fold in fold_scores]
                ensemble_scores[feature_name] = float(np.mean(scores))
            
            return ensemble_scores
            
        except Exception as e:
            self.logger.warning(f"Ensemble scores CV calculation failed: {e}")
            return self._calculate_fold_ensemble_scores(X, y, X, y, feature_names, is_classification)
    
    def _calculate_ensemble_scores_with_stability(self, X: np.ndarray, y: np.ndarray,
                                                feature_names: List[str], is_classification: bool,
                                                groups: Optional[np.ndarray] = None,
                                                n_bootstrap: int = 3) -> Dict[str, Any]:
        """Calculate ensemble scores with stability tracking."""
        try:
            # Calculate base ensemble scores
            ensemble_scores = self._calculate_ensemble_scores_cv(X, y, feature_names, is_classification, groups)
            
            # Calculate stability counts
            stability_counts = {}
            for _ in range(n_bootstrap):
                # Bootstrap sample
                n_samples = X.shape[0]
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]
                
                # Calculate ensemble scores for bootstrap
                bootstrap_scores = self._calculate_ensemble_scores_cv(
                    X_bootstrap, y_bootstrap, feature_names, is_classification, groups
                )
                
                # Track top features
                sorted_features = sorted(bootstrap_scores.items(), key=lambda x: x[1], reverse=True)
                top_count = max(1, len(feature_names) // 4)  # Top 25%
                
                for feature_name, _ in sorted_features[:top_count]:
                    stability_counts[feature_name] = stability_counts.get(feature_name, 0) + 1
            
            return {
                'scores': ensemble_scores,
                'stability_counts': stability_counts
            }
            
        except Exception as e:
            self.logger.warning(f"Ensemble scores with stability calculation failed: {e}")
            return {
                'scores': self._calculate_ensemble_scores_cv(X, y, feature_names, is_classification, groups),
                'stability_counts': {}
            }
    
    def _calculate_fold_ensemble_scores(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: np.ndarray, y_val: np.ndarray,
                                      feature_names: List[str], is_classification: bool) -> Dict[str, float]:
        """Calculate ensemble scores for a single fold."""
        try:
            # LGBM SHAP scores
            lgb_scores = self._calculate_lgb_shap_scores(X_train, y_train, X_val, y_val, feature_names, is_classification)
            
            # LASSO scores
            lasso_scores = self._calculate_lasso_scores(X_train, y_train, X_val, y_val, feature_names, is_classification)
            
            # RandomForest scores
            rf_scores = self._calculate_rf_scores(X_train, y_train, X_val, y_val, feature_names, is_classification)
            
            # Combine scores with z-score normalization
            ensemble_scores = self._combine_scores_with_zscore(
                lgb_scores, lasso_scores, rf_scores, feature_names
            )
            
            return ensemble_scores
            
        except Exception as e:
            self.logger.warning(f"Fold ensemble scores calculation failed: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _calculate_lgb_shap_scores(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 feature_names: List[str], is_classification: bool) -> Dict[str, float]:
        """Calculate LGBM SHAP-based importance scores."""
        if not LIGHTGBM_AVAILABLE or not SHAP_AVAILABLE:
            return {name: 0.0 for name in feature_names}
        
        try:
            # Configure LGBM
            lgb_params = self.config.get('lgb_params', {}).copy()
            lgb_params['random_state'] = self.config.get('random_state', 42)
            
            if is_classification:
                model = lgb.LGBMClassifier(**lgb_params)
            else:
                model = lgb.LGBMRegressor(**lgb_params)
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            
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
            return {name: 0.0 for name in feature_names}
    
    def _calculate_lasso_scores(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              feature_names: List[str], is_classification: bool) -> Dict[str, float]:
        """Calculate LASSO-based importance scores."""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Fit LASSO with cross-validation
            if is_classification:
                lasso = LassoCV(cv=3, random_state=self.config.get('random_state', 42))
                # For classification, use logistic regression
                from sklearn.linear_model import LogisticRegressionCV
                model = LogisticRegressionCV(cv=3, random_state=self.config.get('random_state', 42), max_iter=1000)
                model.fit(X_train_scaled, y_train)
                coefficients = np.abs(model.coef_[0])
            else:
                lasso = LassoCV(cv=3, random_state=self.config.get('random_state', 42))
                lasso.fit(X_train_scaled, y_train)
                coefficients = np.abs(lasso.coef_)
            
            # Create scores dictionary
            lasso_scores = {}
            for i, feature_name in enumerate(feature_names):
                lasso_scores[feature_name] = float(coefficients[i])
            
            return lasso_scores
            
        except Exception as e:
            self.logger.warning(f"LASSO calculation failed: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _calculate_rf_scores(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           feature_names: List[str], is_classification: bool) -> Dict[str, float]:
        """Calculate RandomForest-based importance scores."""
        try:
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    n_jobs=1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42),
                    n_jobs=1
                )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Get feature importance
            importance = model.feature_importances_
            
            # Create scores dictionary
            rf_scores = {}
            for i, feature_name in enumerate(feature_names):
                rf_scores[feature_name] = float(importance[i])
            
            return rf_scores
            
        except Exception as e:
            self.logger.warning(f"RandomForest calculation failed: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _combine_scores_with_zscore(self, lgb_scores: Dict[str, float],
                                  lasso_scores: Dict[str, float],
                                  rf_scores: Dict[str, float],
                                  feature_names: List[str]) -> Dict[str, float]:
        """Combine scores with z-score normalization."""
        try:
            # Extract scores in order
            lgb_values = np.array([lgb_scores.get(name, 0.0) for name in feature_names])
            lasso_values = np.array([lasso_scores.get(name, 0.0) for name in feature_names])
            rf_values = np.array([rf_scores.get(name, 0.0) for name in feature_names])
            
            # Rank scores (descending order)
            lgb_ranks = rankdata(-lgb_values, method='dense')
            lasso_ranks = rankdata(-lasso_values, method='dense')
            rf_ranks = rankdata(-rf_values, method='dense')
            
            # Z-score normalize ranks
            lgb_ranks_z = zscore(lgb_ranks)
            lasso_ranks_z = zscore(lasso_ranks)
            rf_ranks_z = zscore(rf_ranks)
            
            # Average z-scores across models
            combined_scores = (lgb_ranks_z + lasso_ranks_z + rf_ranks_z) / 3
            
            # Convert back to dictionary
            ensemble_scores = {}
            for i, feature_name in enumerate(feature_names):
                ensemble_scores[feature_name] = float(combined_scores[i])
            
            return ensemble_scores
            
        except Exception as e:
            self.logger.warning(f"Score combination with z-score failed: {e}")
            # Fallback to simple average
            ensemble_scores = {}
            for i, feature_name in enumerate(feature_names):
                lgb_score = lgb_scores.get(feature_name, 0.0)
                lasso_score = lasso_scores.get(feature_name, 0.0)
                rf_score = rf_scores.get(feature_name, 0.0)
                ensemble_scores[feature_name] = (lgb_score + lasso_score + rf_score) / 3
            
            return ensemble_scores
    
    def _create_cv_splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation splits."""
        cv_strategy = self.config.get('cv_strategy', 'stratified')
        cv_folds = self.config.get('cv_folds', 5)
        random_state = self.config.get('random_state', 42)
        
        if cv_strategy == 'grouped' and groups is not None:
            cv = GroupKFold(n_splits=cv_folds)
            return list(cv.split(X, y, groups))
        elif cv_strategy == 'stratified':
            # Determine if classification
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))
            if is_classification:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:  # kfold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        return list(cv.split(X, y))
    
    def _calculate_cv_score(self, X: np.ndarray, y: np.ndarray,
                          is_classification: bool, groups: Optional[np.ndarray] = None) -> float:
        """Calculate cross-validation score."""
        try:
            cv_splits = self._create_cv_splits(X, y, groups)
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=self.config.get('random_state', 42))
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=self.config.get('random_state', 42))
                scoring = 'r2'
            
            scores = cross_val_score(
                model, X, y,
                cv=cv_splits,
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
            
            # Apply subsequent stage masks
            if len(stage2_mask) <= len(final_mask):
                stage2_indices = np.where(stage2_mask)[0]
                final_mask = np.zeros_like(final_mask)
                final_mask[stage2_indices] = True
            
            if len(stage3_mask) <= len(final_mask):
                stage3_indices = np.where(stage3_mask)[0]
                final_mask = np.zeros_like(final_mask)
                final_mask[stage3_indices] = True
            
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
            stats['stability_lock_rate'] = stats['stability_locks'] / stats['total_runs']
        else:
            stats['stage1_success_rate'] = 0.0
            stats['stage2_success_rate'] = 0.0
            stats['stage3_success_rate'] = 0.0
            stats['stage4_success_rate'] = 0.0
            stats['plateau_detection_rate'] = 0.0
            stats['stability_lock_rate'] = 0.0
        
        return stats

class PlateauDetector:
    """Enhanced plateau detector with early stopping."""
    
    def __init__(self, threshold: float = 0.002, patience: int = 2):
        """Initialize plateau detector."""
        self.threshold = threshold
        self.patience = patience
        self.scores_history = []
        self.best_score = -np.inf
        self.no_improvement = 0
    
    def check_plateau(self, current_score: float) -> bool:
        """Check if performance has plateaued."""
        self.scores_history.append(current_score)
        
        if len(self.scores_history) < 2:
            return False
        
        # Check for improvement
        if current_score > self.best_score + self.threshold:
            self.best_score = current_score
            self.no_improvement = 0
            return False
        else:
            self.no_improvement += 1
            return self.no_improvement >= self.patience
    
    def reset(self):
        """Reset plateau detector."""
        self.scores_history = []
        self.best_score = -np.inf
        self.no_improvement = 0

def create_enhanced_multi_stage_rfe(config: Optional[Dict[str, Any]] = None) -> EnhancedMultiStageRFE:
    """Create an enhanced multi-stage RFE selector."""
    return EnhancedMultiStageRFE(config)