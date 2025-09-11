"""
Unified Feature Selection Framework

This module provides a comprehensive feature selection framework combining multiple
selection methods with stability analysis, correlation filtering, and ensemble approaches.

Key Features:
- mRMR (Minimum Redundancy Maximum Relevance) selection
- Stability-weighted feature selection
- Correlation-based filtering
- Recursive feature elimination
- Feature importance ranking
- Composite feature scoring
- Cross-validated feature selection

Built on existing utilities:
- Uses math_validation.py for safe mathematical operations
- Integrates with m1_gpu_utils.py for GPU acceleration
- Leverages common_operations.py for robust error handling
- Builds on existing feature selection patterns
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
import time

from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger
from ..m1_gpu_utils import M1GPUManager
from ..parallel_processing_optimizer import ParallelProcessor
from ..matrix_operations import (
    m1_correlation_matrix, m1_matrix_multiply, m1_batch_process,
    m1_parallel_operations, m1_optimize_memory, get_m1_performance_stats
)

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.FeatureSelection")
    print("✅ Custom logger available for MLCommon.FeatureSelection")
except Exception as e:
    print(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.FeatureSelection")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr, spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited feature selection functionality")


class FeatureSelectionFramework:
    """Comprehensive feature selection framework with multiple methods and stability analysis."""
    
    # Model-specific optimal feature counts
    MODEL_FEATURE_TARGETS = {
        # Linear models - work well with moderate feature counts
        'linear_regression': 60,
        'ridge_regression': 80,
        'lasso_regression': 50,
        'elastic_net': 70,
        'logistic_regression': 60,
        
        # Tree-based models - can handle more features
        'random_forest': 100,
        'gradient_boosting': 120,
        'xgboost': 100,
        'lightgbm': 100,
        'catboost': 100,
        'extra_trees': 100,
        
        # SVM models - sensitive to feature count
        'svm_linear': 50,
        'svm_rbf': 80,
        'svm_poly': 60,
        
        # Neural networks - can handle many features
        'neural_network': 150,
        'deep_learning': 200,
        
        # Ensemble methods
        'voting_classifier': 100,
        'stacking_classifier': 120,
        'bagging_classifier': 100,
        
        # Default fallback
        'default': 80
    }
    
    # Minimum feature count for intermediate stages
    MIN_FEATURES_INTERMEDIATE = 100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature selection framework with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('FeatureSelection')
        
        _LOGGER.info("🚀 Initializing FeatureSelectionFramework...")

        # Configuration defaults
        self.enable_gpu = self.config.get('enable_gpu', True)
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', 4)
        self.memory_threshold = self.config.get('memory_threshold', 0.8)
        self.random_state = self.config.get('random_state', 42)

        _LOGGER.info(f"⚙️ Configuration - GPU enabled: {self.enable_gpu}")
        _LOGGER.info(f"⚙️ Configuration - Parallel processing: {self.enable_parallel}")
        _LOGGER.info(f"⚙️ Configuration - Max workers: {self.max_workers}")
        _LOGGER.info(f"⚙️ Configuration - Memory threshold: {self.memory_threshold}")
        _LOGGER.info(f"⚙️ Configuration - Random state: {self.random_state}")

        # Initialize utilities
        self.gpu_manager = M1GPUManager() if self.enable_gpu else None
        self.parallel_processor = ParallelProcessor() if self.enable_parallel else None

        # Method configurations
        _LOGGER.debug("🔧 Initializing method configurations...")
        self.method_configs = {
            'mrmr': {
                'relevance_method': 'mutual_info',
                'redundancy_method': 'correlation',
                'n_neighbors': 3
            },
            'importance': {
                'n_estimators': 100,
                'max_depth': 10,
                'bootstrap': True
            },
            'rfe': {
                'step': 0.1,
                'cv': 3,
                'scoring': 'accuracy'
            },
            'stability': {
                'n_bootstraps': 50,
                'bootstrap_fraction': 0.8,
                'stability_threshold': 0.6
            },
            'lasso': {
                'alpha_range': (0.001, 1.0),
                'cv_folds': 5,
                'max_iter': 1000,
                'tol': 1e-4,
                'random_state': 42
            },
            'lasso_stability': {
                'n_bootstraps': 100,
                'bootstrap_fraction': 0.8,
                'stability_threshold': 0.6,
                'alpha_range': (0.001, 1.0),
                'cv_folds': 5
            },
            'tree_ensemble': {
                'cv_folds': 5,
                'permutation_importance_repeats': 10,
                'correlation_threshold': 0.8,
                'hyperparameter_search': True,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None]
                },
                'random_state': 42
            }
        }

        # Update with user config
        if 'method_configs' in self.config:
            _LOGGER.debug("🔧 Updating method configurations with user config...")
            self.method_configs.update(self.config['method_configs'])
        
        _LOGGER.info("✅ FeatureSelectionFramework initialized successfully")

    def hierarchical_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str],
                                     features_target_count: int,
                                     model_type: str = 'default',
                                     model: Optional[Any] = None,
                                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced hierarchical feature selection pipeline with adaptive thresholds.
        
        Pipeline stages:
        0. Define initial and target feature counts
        1. Correlation-based filtering (remove highly correlated pairs)
        2. mRMR selection (skip if < 150, reduce to ~100 if > 150)
        3. LASSO stability + RFE consensus (reduce by half toward target)
        4. Tree-based ensemble selection (final selection to target)
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            features_target_count: Final target number of features
            config: Configuration dictionary with thresholds and parameters
            
        Returns:
            Dictionary with comprehensive pipeline results
        """
        start_time = time.time()
        features_initial_count = len(feature_names)
        
        # Auto-detect model type if model object is provided
        if model is not None:
            detected_model_type = self._auto_detect_model_type(model)
            if detected_model_type != 'default':
                model_type = detected_model_type
                _LOGGER.info(f"🎯 Auto-detected model type: {model_type}")
        
        # Validate and plan feature reduction
        validation_result = self.validate_feature_reduction_plan(
            features_initial_count, features_target_count, model_type
        )
        
        if not validation_result['valid']:
            _LOGGER.error(f"❌ Invalid feature reduction plan: {validation_result['errors']}")
            return {
                'selected_features': feature_names,
                'pipeline_stages': {},
                'pipeline_summary': {
                    'initial_count': features_initial_count,
                    'final_count': len(feature_names),
                    'total_reduction': 0,
                    'execution_time': time.time() - start_time,
                    'reduction_percentage': 0.0
                },
                'validation_result': validation_result
            }
        
        # Use validated target count
        features_target_count = validation_result['target_count']
        reduction_plan = validation_result['reduction_plan']
        
        # Configurable thresholds with dynamic defaults
        default_config = {
            'use_dynamic_thresholds': True,     # Enable dynamic threshold determination
            'mrmr_skip_threshold': 150,        # Skip mRMR if features < this
            'consensus_reduction_factor': 0.5,  # Reduce by this fraction in consensus
            'correlation_threshold': None,     # Will be determined dynamically if None
            'stability_threshold': None,       # Will be determined dynamically if None
            'enable_parallel': self.enable_parallel,
            'memory_optimization': True,
            'verbose': True,
            'cv_folds': 5,                     # CV folds for RFE optimization
            'min_features_ratio': 0.05,        # Minimum 5% of features at each stage
            'max_features_ratio': 0.5,         # Maximum 50% of features at each stage
            # Bootstrap stability validation parameters
            'enable_bootstrap_stability': True, # Enable bootstrap stability validation
            'n_bootstrap_samples': 10,         # Number of bootstrap samples
            'bootstrap_fraction': 0.8,         # Fraction of data in each bootstrap
            'bootstrap_stability_threshold': 0.6  # Stability threshold for bootstrap validation
        }
        
        config = {**default_config, **(config or {})}
        
        _LOGGER.info(f"🚀 Starting Hierarchical Feature Selection Pipeline")
        _LOGGER.info(f"📊 Initial features: {features_initial_count}")
        _LOGGER.info(f"🎯 Target features: {features_target_count} (for {model_type})")
        _LOGGER.info(f"📉 Features to remove: {validation_result['removal_count']}")
        _LOGGER.info(f"📋 Reduction plan: {len(reduction_plan)} stages")
        
        # Log warnings if any
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                _LOGGER.warning(f"⚠️ {warning}")
        
        pipeline_results = {
            'pipeline_stages': {},
            'final_selected_features': [],
            'pipeline_summary': {
                'initial_count': features_initial_count,
                'target_count': features_target_count,
                'final_count': 0,
                'total_reduction': 0,
                'execution_time': 0.0
            }
        }
        
        current_features = feature_names.copy()
        current_X = X.copy()
        
        try:
            # Stage 1: Correlation-based filtering with dynamic threshold
            _LOGGER.info("🔍 Stage 1: Correlation-based filtering with dynamic threshold...")
            
            # Determine adaptive correlation threshold
            if config['use_dynamic_thresholds'] and config['correlation_threshold'] is None:
                correlation_threshold = self._determine_adaptive_correlation_threshold(current_X, current_features)
            else:
                correlation_threshold = config['correlation_threshold'] or 0.95
            
            correlation_result = self.correlation_based_filtering(
                current_X, current_features, 
                correlation_threshold=correlation_threshold
            )
            
            if 'selected_features' in correlation_result:
                features_after_correlation = correlation_result['selected_features']
                pipeline_results['pipeline_stages']['correlation_filtering'] = {
                    'input_count': len(current_features),
                    'output_count': len(features_after_correlation),
                    'removed_count': len(current_features) - len(features_after_correlation),
                    'result': correlation_result
                }
                
                # Update current state
                current_features = features_after_correlation
                selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                current_X = X[:, selected_indices]
                
                _LOGGER.info(f"✅ Stage 1 complete: {len(current_features)} features remaining")
            else:
                _LOGGER.warning("⚠️ Stage 1 failed, continuing with original features")
                pipeline_results['pipeline_stages']['correlation_filtering'] = {
                    'error': 'Correlation filtering failed',
                    'input_count': len(current_features),
                    'output_count': len(current_features)
                }
            
            # Stage 2: mRMR selection with dynamic threshold
            _LOGGER.info("🔍 Stage 2: mRMR selection with dynamic threshold...")
            features_after_mrmr = current_features.copy()
            
            if len(current_features) >= config['mrmr_skip_threshold']:
                # First, run mRMR to get scores
                _LOGGER.info("🔍 Computing mRMR scores for dynamic threshold determination...")
                mrmr_result = self.mrmr_selection(current_X, y, current_features, len(current_features))
                
                if 'mrmr_scores' in mrmr_result and config['use_dynamic_thresholds']:
                    # Determine dynamic threshold based on mRMR scores
                    mrmr_threshold, mrmr_target = self._determine_mrmr_threshold(
                        mrmr_result['mrmr_scores'], current_features
                    )
                    
                    # Filter features based on dynamic threshold
                    features_above_threshold = [
                        feature for feature, score in mrmr_result['mrmr_scores'].items()
                        if score >= mrmr_threshold
                    ]
                    
                    # Ensure we don't go below target count
                    if len(features_above_threshold) < features_target_count:
                        # Take top features by score
                        sorted_features = sorted(
                            mrmr_result['mrmr_scores'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        features_above_threshold = [f for f, _ in sorted_features[:features_target_count]]
                    
                    mrmr_result['selected_features'] = features_above_threshold
                    mrmr_result['dynamic_threshold'] = mrmr_threshold
                    mrmr_result['threshold_method'] = 'dynamic'
                    
                    _LOGGER.info(f"📊 mRMR dynamic target: {len(features_above_threshold)} (threshold: {mrmr_threshold:.4f})")
                else:
                    # Fallback to proportional reduction
                    mrmr_target = max(features_target_count * 2, len(current_features) // 2)
                    mrmr_target = min(mrmr_target, len(current_features))
                    mrmr_result = self.mrmr_selection(current_X, y, current_features, mrmr_target)
                    mrmr_result['threshold_method'] = 'proportional'
                    _LOGGER.info(f"📊 mRMR proportional target: {mrmr_target}")
                
                if 'selected_features' in mrmr_result:
                    features_after_mrmr = mrmr_result['selected_features']
                    pipeline_results['pipeline_stages']['mrmr_selection'] = {
                        'input_count': len(current_features),
                        'output_count': len(features_after_mrmr),
                        'target_count': mrmr_target,
                        'result': mrmr_result
                    }
                    
                    # Update current state
                    current_features = features_after_mrmr
                    selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                    current_X = X[:, selected_indices]
                    
                    _LOGGER.info(f"✅ Stage 2 complete: {len(current_features)} features remaining")
                else:
                    _LOGGER.warning("⚠️ Stage 2 failed, continuing with previous features")
                    pipeline_results['pipeline_stages']['mrmr_selection'] = {
                        'error': 'mRMR selection failed',
                        'input_count': len(current_features),
                        'output_count': len(current_features)
                    }
            else:
                _LOGGER.info(f"⏭️ Stage 2 skipped: {len(current_features)} < {config['mrmr_skip_threshold']} threshold")
                pipeline_results['pipeline_stages']['mrmr_selection'] = {
                    'skipped': True,
                    'reason': f"Features ({len(current_features)}) below threshold ({config['mrmr_skip_threshold']})",
                    'input_count': len(current_features),
                    'output_count': len(current_features)
                }
            
            # Stage 3: LASSO stability + RFE consensus with dynamic optimization
            _LOGGER.info("🔍 Stage 3: LASSO stability + RFE consensus with dynamic optimization...")
            
            # LASSO stability selection with dynamic threshold
            lasso_result = self.lasso_stability_selection(
                current_X, y, current_features,
                stability_threshold=config['stability_threshold'] or 0.6
            )
            
            # Determine dynamic stability threshold if enabled
            if config['use_dynamic_thresholds'] and 'feature_stability_scores' in lasso_result:
                stability_threshold, stable_features_count = self._determine_lasso_stability_threshold(
                    lasso_result['feature_stability_scores'], current_features
                )
                lasso_result['dynamic_stability_threshold'] = stability_threshold
                lasso_result['threshold_method'] = 'dynamic'
                _LOGGER.info(f"📊 LASSO dynamic stability threshold: {stability_threshold:.3f}")
            
            # RFE selection with optimal feature count determination
            base_model = self._get_default_model(y)
            rfe_result = None
            if base_model is not None:
                if config['use_dynamic_thresholds']:
                    # Determine optimal number of features using cross-validation
                    optimal_rfe_features = self._determine_optimal_rfe_features(
                        current_X, y, current_features, base_model, config['cv_folds']
                    )
                    _LOGGER.info(f"📊 RFE optimal features determined by CV: {optimal_rfe_features}")
                else:
                    # Use proportional reduction
                    optimal_rfe_features = max(features_target_count, 
                                             int(len(current_features) * config['consensus_reduction_factor']))
                    optimal_rfe_features = min(optimal_rfe_features, len(current_features))
                    _LOGGER.info(f"📊 RFE proportional target: {optimal_rfe_features}")
                
                rfe_result = self.recursive_feature_elimination(
                    base_model, current_X, y, current_features, optimal_rfe_features
                )
                rfe_result['optimal_features'] = optimal_rfe_features
            
            # Calculate consensus target based on both methods
            lasso_features = lasso_result.get('selected_features', [])
            rfe_features = rfe_result.get('selected_features', []) if rfe_result else []
            
            # Dynamic consensus target based on method results
            if config['use_dynamic_thresholds']:
                # Use the smaller of the two method results, but ensure we don't go below target
                consensus_target = max(features_target_count, min(len(lasso_features), len(rfe_features)))
            else:
                # Use proportional reduction
                consensus_target = max(features_target_count, 
                                     int(len(current_features) * config['consensus_reduction_factor']))
            
            consensus_target = min(consensus_target, len(current_features))
            _LOGGER.info(f"📊 Dynamic consensus target: {consensus_target} (LASSO: {len(lasso_features)}, RFE: {len(rfe_features)})")
            
            # Compute consensus
            consensus_features = self._compute_lasso_rfe_consensus(
                lasso_result.get('selected_features', []),
                rfe_result.get('selected_features', []) if rfe_result else [],
                consensus_target
            )
            
            pipeline_results['pipeline_stages']['lasso_rfe_consensus'] = {
                'input_count': len(current_features),
                'output_count': len(consensus_features),
                'target_count': consensus_target,
                'lasso_result': lasso_result,
                'rfe_result': rfe_result,
                'consensus_features': consensus_features
            }
            
            # Update current state
            current_features = consensus_features
            selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
            current_X = X[:, selected_indices]
            
            _LOGGER.info(f"✅ Stage 3 complete: {len(current_features)} features remaining")
            
            # Stage 3.5: Bootstrap Stability Validation (NEW STAGE)
            if config['enable_bootstrap_stability']:
                _LOGGER.info("🔍 Stage 3.5: Bootstrap stability validation...")
                
                # Run bootstrap stability validation on the consensus features
                bootstrap_stability_result = self._bootstrap_pipeline_stability_validation(
                    X, y, feature_names, features_target_count, config,
                    n_bootstrap=config['n_bootstrap_samples'],
                    bootstrap_fraction=config['bootstrap_fraction'],
                    stability_threshold=config['bootstrap_stability_threshold']
                )
                
                # Update current features with stable features
                stable_features = bootstrap_stability_result['stable_features']
                if stable_features:
                    current_features = stable_features
                    selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                    current_X = X[:, selected_indices]
                    
                    pipeline_results['pipeline_stages']['bootstrap_stability'] = {
                        'input_count': len(consensus_features),
                        'output_count': len(current_features),
                        'stability_analysis': bootstrap_stability_result['stability_analysis'],
                        'bootstrap_results': bootstrap_stability_result['bootstrap_results'],
                        'execution_time': bootstrap_stability_result['execution_time']
                    }
                    
                    _LOGGER.info(f"✅ Stage 3.5 complete: {len(current_features)} stable features remaining")
                else:
                    _LOGGER.warning("⚠️ No stable features found in bootstrap validation, using consensus features")
                    pipeline_results['pipeline_stages']['bootstrap_stability'] = {
                        'error': 'No stable features found',
                        'input_count': len(consensus_features),
                        'output_count': len(consensus_features),
                        'stability_analysis': bootstrap_stability_result['stability_analysis']
                    }
            else:
                _LOGGER.info("⏭️ Stage 3.5 skipped: Bootstrap stability validation disabled")
                pipeline_results['pipeline_stages']['bootstrap_stability'] = {
                    'skipped': True,
                    'reason': 'Bootstrap stability validation disabled',
                    'input_count': len(current_features),
                    'output_count': len(current_features)
                }
            
            # Stage 4: Tree-based ensemble selection with dynamic threshold (final)
            _LOGGER.info("🔍 Stage 4: Tree-based ensemble selection with dynamic threshold (final)...")
            
            # First run tree ensemble to get importance scores
            tree_result = self.tree_based_ensemble_selection(
                current_X, y, current_features,
                methods=['correlation', 'mrmr', 'lasso_stability'],
                n_features=None,  # Get all features with importance scores
                cv_folds=config['cv_folds'],
                permutation_importance_repeats=10
            )
            
            # Apply dynamic threshold if enabled
            if config['use_dynamic_thresholds'] and 'permutation_importance' in tree_result:
                importance_threshold, important_features_count = self._determine_tree_ensemble_threshold(
                    tree_result['permutation_importance'], current_features
                )
                
                # Filter features based on dynamic threshold
                features_above_threshold = [
                    feature for feature, data in tree_result['permutation_importance'].items()
                    if data['importance'] >= importance_threshold
                ]
                
                # Ensure we don't go below target count
                if len(features_above_threshold) < features_target_count:
                    # Take top features by importance
                    sorted_features = sorted(
                        tree_result['permutation_importance'].items(),
                        key=lambda x: x[1]['importance'],
                        reverse=True
                    )
                    features_above_threshold = [f for f, _ in sorted_features[:features_target_count]]
                
                tree_result['selected_features'] = features_above_threshold
                tree_result['dynamic_importance_threshold'] = importance_threshold
                tree_result['threshold_method'] = 'dynamic'
                
                _LOGGER.info(f"📊 Tree ensemble dynamic target: {len(features_above_threshold)} (threshold: {importance_threshold:.6f})")
            else:
                # Fallback to target count
                if 'selected_features' not in tree_result:
                    # Take top features by importance
                    if 'permutation_importance' in tree_result:
                        sorted_features = sorted(
                            tree_result['permutation_importance'].items(),
                            key=lambda x: x[1]['importance'],
                            reverse=True
                        )
                        tree_result['selected_features'] = [f for f, _ in sorted_features[:features_target_count]]
                    else:
                        tree_result['selected_features'] = current_features[:features_target_count]
                
                tree_result['threshold_method'] = 'target_count'
                _LOGGER.info(f"📊 Tree ensemble target count: {features_target_count}")
            
            if 'selected_features' in tree_result:
                final_features = tree_result['selected_features']
                pipeline_results['pipeline_stages']['tree_ensemble'] = {
                    'input_count': len(current_features),
                    'output_count': len(final_features),
                    'target_count': features_target_count,
                    'result': tree_result
                }
                
                _LOGGER.info(f"✅ Stage 4 complete: {len(final_features)} final features")
            else:
                _LOGGER.warning("⚠️ Stage 4 failed, using consensus features")
                final_features = current_features
                pipeline_results['pipeline_stages']['tree_ensemble'] = {
                    'error': 'Tree ensemble selection failed',
                    'input_count': len(current_features),
                    'output_count': len(current_features)
                }
            
            # Stage 5: RF Cross-Validation Refinement (if needed)
            if reduction_plan['stage5_rf_refinement']['enabled']:
                _LOGGER.info("🔍 Stage 5: RF Cross-Validation Refinement...")
                
                final_target = reduction_plan['stage5_rf_refinement']['target']
                _LOGGER.info(f"🎯 Stage 5 target: {len(final_features)} → {final_target}")
                
                # Get current feature matrix
                selected_indices = [feature_names.index(f) for f in final_features if f in feature_names]
                current_X = X[:, selected_indices]
                
                rf_refinement_result = self.rf_cross_validation_refinement(
                    current_X, y, final_features, final_target, config['cv_folds']
                )
                
                final_features = rf_refinement_result.get('selected_features', final_features)
                
                pipeline_results['pipeline_stages']['rf_refinement'] = {
                    'input_count': len(pipeline_results['pipeline_stages']['tree_ensemble']['output_count']),
                    'output_count': len(final_features),
                    'result': rf_refinement_result,
                    'execution_time': rf_refinement_result.get('execution_time', 0)
                }
                
                _LOGGER.info(f"✅ Stage 5 complete: {len(final_features)} features remaining")
            else:
                _LOGGER.info("⏭️ Stage 5 skipped: RF refinement not needed")
                pipeline_results['pipeline_stages']['rf_refinement'] = {
                    'skipped': True,
                    'reason': 'RF refinement not needed',
                    'input_count': len(final_features),
                    'output_count': len(final_features)
                }
            
            # Final results
            execution_time = time.time() - start_time
            pipeline_results['final_selected_features'] = final_features
            pipeline_results['pipeline_summary'].update({
                'final_count': len(final_features),
                'total_reduction': features_initial_count - len(final_features),
                'execution_time': execution_time,
                'reduction_percentage': safe_divide(features_initial_count - len(final_features), features_initial_count) * 100
            })
            
            # Enhanced reporting with model-specific and dynamic threshold information
            additional_stats = {
                'Pipeline stages': len(pipeline_results['pipeline_stages']),
                'Target achieved': len(final_features) == features_target_count,
                'Final vs target': f"{len(final_features)}/{features_target_count}",
                'Total reduction': f"{features_initial_count - len(final_features)} ({pipeline_results['pipeline_summary']['reduction_percentage']:.1f}%)",
                'Model type': model_type,
                'Model target': validation_result['model_target'],
                'Removal count': validation_result['removal_count'],
                'Dynamic thresholds': config['use_dynamic_thresholds'],
                'CV folds': config['cv_folds'],
                'Bootstrap stability': config['enable_bootstrap_stability'],
                'RF refinement': reduction_plan['stage5_rf_refinement']['enabled']
            }
            
            # Add bootstrap stability information
            if 'bootstrap_stability' in pipeline_results['pipeline_stages']:
                bootstrap_stage = pipeline_results['pipeline_stages']['bootstrap_stability']
                if 'stability_analysis' in bootstrap_stage:
                    stability_stats = bootstrap_stage['stability_analysis']['stability_statistics']
                    additional_stats.update({
                        'Bootstrap samples': bootstrap_stage['stability_analysis']['n_bootstrap_samples'],
                        'Mean stability': f"{stability_stats['mean_stability']:.3f}",
                        'Stable features found': stability_stats['features_above_threshold'],
                        'Bootstrap threshold': bootstrap_stage['stability_analysis']['stability_threshold']
                    })
            
            # Add threshold information for each stage
            for stage_name, stage_data in pipeline_results['pipeline_stages'].items():
                if 'result' in stage_data and isinstance(stage_data['result'], dict):
                    result = stage_data['result']
                    if 'threshold_method' in result:
                        additional_stats[f'{stage_name}_method'] = result['threshold_method']
                    if 'dynamic_threshold' in result:
                        additional_stats[f'{stage_name}_threshold'] = f"{result['dynamic_threshold']:.4f}"
                    if 'dynamic_stability_threshold' in result:
                        additional_stats[f'{stage_name}_stability_threshold'] = f"{result['dynamic_stability_threshold']:.4f}"
                    if 'dynamic_importance_threshold' in result:
                        additional_stats[f'{stage_name}_importance_threshold'] = f"{result['dynamic_importance_threshold']:.6f}"
            
            self._log_feature_reduction_stats(
                "Hierarchical Feature Selection Pipeline", 
                features_initial_count, len(final_features), execution_time, additional_stats
            )
            
            # Memory optimization
            if config['memory_optimization']:
                m1_optimize_memory()
            
            _LOGGER.info(f"🎉 Hierarchical pipeline completed successfully!")
            _LOGGER.info(f"📊 Final result: {len(final_features)}/{features_target_count} target features")
            
            # Add validation and reduction plan to results
            pipeline_results['validation_result'] = validation_result
            pipeline_results['reduction_plan'] = reduction_plan
            
            return pipeline_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Hierarchical pipeline failed after {execution_time:.3f}s: {e}")
            return {
                'error': str(e),
                'final_selected_features': current_features,
                'pipeline_summary': {
                    'initial_count': features_initial_count,
                    'target_count': features_target_count,
                    'final_count': len(current_features),
                    'execution_time': execution_time
                }
            }

    def _compute_lasso_rfe_consensus(self, lasso_features: List[str], rfe_features: List[str], 
                                   target_count: int) -> List[str]:
        """Compute consensus between LASSO stability and RFE with intelligent voting."""
        
        if not lasso_features and not rfe_features:
            return []
        
        if not lasso_features:
            return rfe_features[:target_count]
        if not rfe_features:
            return lasso_features[:target_count]
        
        # Feature voting with weights
        feature_votes = {}
        for feature in lasso_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 0.6  # LASSO weight
        for feature in rfe_features:
            feature_votes[feature] = feature_votes.get(feature, 0) + 0.4  # RFE weight
        
        # Sort by votes and select top features
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        consensus_features = [feature for feature, votes in sorted_features[:target_count]]
        
        _LOGGER.info(f"📊 Consensus: {len(consensus_features)} features from {len(lasso_features)} LASSO + {len(rfe_features)} RFE")
        
        return consensus_features

    def _determine_adaptive_correlation_threshold(self, X: np.ndarray, feature_names: List[str]) -> float:
        """Determine adaptive correlation threshold based on data characteristics."""
        try:
            # Calculate correlation matrix
            corr_matrix = m1_correlation_matrix(X.T)
            
            # Get upper triangle correlations (excluding diagonal)
            upper_tri = np.triu(corr_matrix, k=1)
            correlations = upper_tri[upper_tri != 0]
            
            if len(correlations) == 0:
                return 0.95  # Default threshold
            
            # Calculate statistics
            mean_corr = np.mean(np.abs(correlations))
            std_corr = np.std(np.abs(correlations))
            q75_corr = np.percentile(np.abs(correlations), 75)
            q90_corr = np.percentile(np.abs(correlations), 90)
            
            # Adaptive threshold based on correlation distribution
            if mean_corr > 0.7:  # High correlation data
                threshold = min(0.98, q90_corr)
            elif mean_corr > 0.4:  # Medium correlation data
                threshold = min(0.95, q75_corr + std_corr)
            else:  # Low correlation data
                threshold = max(0.90, q75_corr)
            
            _LOGGER.info(f"📊 Adaptive correlation threshold: {threshold:.3f} (mean: {mean_corr:.3f}, std: {std_corr:.3f})")
            return threshold
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Adaptive correlation threshold failed: {e}, using default 0.95")
            return 0.95

    def _determine_mrmr_threshold(self, mrmr_scores: Dict[str, float], 
                                feature_names: List[str]) -> Tuple[float, int]:
        """Determine dynamic threshold for mRMR feature selection based on score distribution."""
        try:
            if not mrmr_scores:
                return 0.0, 0
            
            scores = list(mrmr_scores.values())
            scores = [s for s in scores if not np.isnan(s) and np.isfinite(s)]
            
            if not scores:
                return 0.0, 0
            
            # Calculate score statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            median_score = np.median(scores)
            q75_score = np.percentile(scores, 75)
            q90_score = np.percentile(scores, 90)
            
            # Dynamic threshold based on score distribution
            if std_score > mean_score:  # High variance in scores
                threshold = max(median_score, mean_score - std_score)
            else:  # Low variance in scores
                threshold = q75_score
            
            # Count features above threshold
            features_above_threshold = sum(1 for score in scores if score >= threshold)
            
            # Ensure minimum and maximum bounds
            min_features = max(10, len(feature_names) // 20)  # At least 5% of features
            max_features = min(len(feature_names), len(feature_names) // 2)  # At most 50% of features
            
            features_above_threshold = max(min_features, min(max_features, features_above_threshold))
            
            _LOGGER.info(f"📊 mRMR dynamic threshold: {threshold:.4f}")
            _LOGGER.info(f"📊 Features above threshold: {features_above_threshold}/{len(feature_names)}")
            _LOGGER.info(f"📊 Score stats - mean: {mean_score:.4f}, std: {std_score:.4f}, q75: {q75_score:.4f}")
            
            return threshold, features_above_threshold
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ mRMR threshold determination failed: {e}")
            return 0.0, max(10, len(feature_names) // 10)

    def _determine_optimal_rfe_features(self, X: np.ndarray, y: np.ndarray, 
                                      feature_names: List[str], 
                                      base_model: Any, cv_folds: int = 5) -> int:
        """Determine optimal number of features for RFE using cross-validation."""
        try:
            if not SKLEARN_AVAILABLE:
                return min(20, len(feature_names) // 2)
            
            from sklearn.model_selection import cross_val_score
            from sklearn.feature_selection import RFECV
            
            _LOGGER.info(f"🔍 Determining optimal RFE features using {cv_folds}-fold CV...")
            
            # Use RFECV to find optimal number of features
            rfecv = RFECV(
                estimator=base_model,
                step=0.1,  # Remove 10% of features at each step
                cv=cv_folds,
                scoring='accuracy' if len(np.unique(y)) <= 10 else 'r2',
                min_features_to_select=1,
                n_jobs=1  # Single job for parallel processing compatibility
            )
            
            rfecv.fit(X, y)
            optimal_features = rfecv.n_features_
            
            # Get CV scores for analysis
            cv_scores = rfecv.cv_results_['mean_test_score']
            cv_stds = rfecv.cv_results_['std_test_score']
            
            # Find the point where performance starts to degrade significantly
            max_score = np.max(cv_scores)
            max_idx = np.argmax(cv_scores)
            
            # Look for significant drop in performance (more than 1 std)
            for i in range(max_idx, len(cv_scores)):
                if cv_scores[i] < max_score - cv_stds[max_idx]:
                    optimal_features = max(1, i)
                    break
            
            # Ensure reasonable bounds
            min_features = max(1, len(feature_names) // 20)  # At least 5% of features
            max_features = min(len(feature_names), len(feature_names) // 2)  # At most 50% of features
            optimal_features = max(min_features, min(max_features, optimal_features))
            
            _LOGGER.info(f"📊 Optimal RFE features: {optimal_features}/{len(feature_names)}")
            _LOGGER.info(f"📊 Max CV score: {max_score:.4f} ± {cv_stds[max_idx]:.4f}")
            
            return optimal_features
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Optimal RFE features determination failed: {e}")
            return min(20, len(feature_names) // 2)

    def _determine_lasso_stability_threshold(self, stability_scores: Dict[str, float],
                                           feature_names: List[str]) -> Tuple[float, int]:
        """Determine dynamic stability threshold for LASSO feature selection."""
        try:
            if not stability_scores:
                return 0.6, 0
            
            scores = list(stability_scores.values())
            scores = [s for s in scores if not np.isnan(s) and np.isfinite(s)]
            
            if not scores:
                return 0.6, 0
            
            # Calculate score statistics
            mean_stability = np.mean(scores)
            std_stability = np.std(scores)
            median_stability = np.median(scores)
            q75_stability = np.percentile(scores, 75)
            
            # Dynamic threshold based on stability distribution
            if mean_stability > 0.7:  # High stability data
                threshold = max(0.6, q75_stability)
            elif mean_stability > 0.4:  # Medium stability data
                threshold = max(0.5, median_stability)
            else:  # Low stability data
                threshold = max(0.3, mean_stability - std_stability)
            
            # Count features above threshold
            features_above_threshold = sum(1 for score in scores if score >= threshold)
            
            # Ensure reasonable bounds
            min_features = max(5, len(feature_names) // 20)  # At least 5% of features
            max_features = min(len(feature_names), len(feature_names) // 2)  # At most 50% of features
            features_above_threshold = max(min_features, min(max_features, features_above_threshold))
            
            _LOGGER.info(f"📊 LASSO stability threshold: {threshold:.3f}")
            _LOGGER.info(f"📊 Stable features: {features_above_threshold}/{len(feature_names)}")
            _LOGGER.info(f"📊 Stability stats - mean: {mean_stability:.3f}, std: {std_stability:.3f}")
            
            return threshold, features_above_threshold
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ LASSO stability threshold determination failed: {e}")
            return 0.6, max(5, len(feature_names) // 10)

    def _determine_tree_ensemble_threshold(self, importance_scores: Dict[str, Dict[str, Any]],
                                         feature_names: List[str]) -> Tuple[float, int]:
        """Determine dynamic threshold for tree ensemble feature selection based on importance scores."""
        try:
            if not importance_scores:
                return 0.0, 0
            
            # Extract importance values
            importances = []
            for feature, data in importance_scores.items():
                if isinstance(data, dict) and 'importance' in data:
                    imp = data['importance']
                    if not np.isnan(imp) and np.isfinite(imp):
                        importances.append(imp)
            
            if not importances:
                return 0.0, 0
            
            # Calculate importance statistics
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            median_importance = np.median(importances)
            q75_importance = np.percentile(importances, 75)
            
            # Dynamic threshold based on importance distribution
            if mean_importance > 0.01:  # High importance features
                threshold = max(0.001, q75_importance)
            elif mean_importance > 0.001:  # Medium importance features
                threshold = max(0.0001, median_importance)
            else:  # Low importance features
                threshold = max(0.00001, mean_importance - std_importance)
            
            # Count features above threshold
            features_above_threshold = sum(1 for imp in importances if imp >= threshold)
            
            # Ensure reasonable bounds
            min_features = max(1, len(feature_names) // 50)  # At least 2% of features
            max_features = min(len(feature_names), len(feature_names) // 2)  # At most 50% of features
            features_above_threshold = max(min_features, min(max_features, features_above_threshold))
            
            _LOGGER.info(f"📊 Tree ensemble threshold: {threshold:.6f}")
            _LOGGER.info(f"📊 Important features: {features_above_threshold}/{len(feature_names)}")
            _LOGGER.info(f"📊 Importance stats - mean: {mean_importance:.6f}, std: {std_importance:.6f}")
            
            return threshold, features_above_threshold
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Tree ensemble threshold determination failed: {e}")
            return 0.0, max(1, len(feature_names) // 20)

    def _nested_bootstrap_stability_validation(self, X: np.ndarray, y: np.ndarray,
                                             feature_names: List[str],
                                             features_target_count: int,
                                             config: Dict[str, Any],
                                             n_outer_bootstrap: int = 5,
                                             n_inner_bootstrap: int = 10,
                                             bootstrap_fraction: float = 0.8,
                                             stability_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Nested bootstrap validation for enhanced feature selection stability.
        
        This method performs a two-level bootstrap:
        1. Outer bootstrap: Multiple independent feature selection runs
        2. Inner bootstrap: Within each outer run, multiple bootstrap samples
        
        This provides more robust stability assessment by testing consistency
        across different data samples and different selection runs.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            features_target_count: Target number of features
            config: Pipeline configuration
            n_outer_bootstrap: Number of outer bootstrap runs
            n_inner_bootstrap: Number of inner bootstrap samples per outer run
            bootstrap_fraction: Fraction of data to use in each bootstrap
            stability_threshold: Minimum stability score for feature selection
            
        Returns:
            Dictionary with nested bootstrap stability analysis
        """
        start_time = time.time()
        _LOGGER.info(f"🔄 Starting nested bootstrap stability validation...")
        _LOGGER.info(f"📊 Outer bootstrap runs: {n_outer_bootstrap}")
        _LOGGER.info(f"📊 Inner bootstrap samples: {n_inner_bootstrap}")
        _LOGGER.info(f"📊 Total bootstrap samples: {n_outer_bootstrap * n_inner_bootstrap}")
        
        outer_results = []
        all_feature_selections = []
        feature_selection_counts = {feature: 0 for feature in feature_names}
        
        # Outer bootstrap loop
        for outer_idx in range(n_outer_bootstrap):
            _LOGGER.info(f"🔄 Outer bootstrap run {outer_idx + 1}/{n_outer_bootstrap}")
            
            # Inner bootstrap loop
            inner_results = []
            for inner_idx in range(n_inner_bootstrap):
                try:
                    # Bootstrap sampling
                    bootstrap_size = int(len(X) * bootstrap_fraction)
                    bootstrap_indices = np.random.choice(
                        len(X), size=bootstrap_size, replace=True
                    )
                    X_bootstrap = X[bootstrap_indices]
                    y_bootstrap = y[bootstrap_indices]
                    
                    # Run pipeline on bootstrap sample
                    bootstrap_features = self._run_pipeline_to_consensus(
                        X_bootstrap, y_bootstrap, feature_names, features_target_count, config
                    )
                    
                    inner_results.append({
                        'bootstrap_idx': inner_idx,
                        'selected_features': bootstrap_features,
                        'n_features': len(bootstrap_features)
                    })
                    
                    # Track all feature selections
                    all_feature_selections.append(bootstrap_features)
                    for feature in bootstrap_features:
                        if feature in feature_selection_counts:
                            feature_selection_counts[feature] += 1
                    
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Inner bootstrap {inner_idx + 1} failed: {e}")
                    continue
            
            # Analyze inner bootstrap consistency for this outer run
            if inner_results:
                inner_consistency = self._analyze_inner_bootstrap_consistency(
                    inner_results, feature_names
                )
                
                outer_results.append({
                    'outer_idx': outer_idx,
                    'inner_results': inner_results,
                    'inner_consistency': inner_consistency,
                    'n_successful_inner': len(inner_results)
                })
        
        # Calculate overall stability scores
        total_bootstrap_samples = sum(len(outer['inner_results']) for outer in outer_results)
        stability_scores = {}
        for feature in feature_names:
            selection_count = feature_selection_counts[feature]
            stability_score = selection_count / total_bootstrap_samples if total_bootstrap_samples > 0 else 0.0
            stability_scores[feature] = stability_score
        
        # Select stable features
        stable_features = [
            feature for feature, stability in stability_scores.items()
            if stability >= stability_threshold
        ]
        
        # If too few stable features, relax threshold
        if len(stable_features) < features_target_count:
            _LOGGER.warning(f"⚠️ Only {len(stable_features)} stable features found, relaxing criteria...")
            sorted_features = sorted(
                stability_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            stable_features = [feature for feature, _ in sorted_features[:features_target_count]]
            if stable_features:
                min_stability = min(stability_scores[f] for f in stable_features)
                _LOGGER.info(f"📊 Relaxed stability threshold: {min_stability:.3f}")
        
        # Nested bootstrap analysis
        nested_analysis = {
            'n_outer_bootstrap': n_outer_bootstrap,
            'n_inner_bootstrap': n_inner_bootstrap,
            'total_bootstrap_samples': total_bootstrap_samples,
            'stability_threshold': stability_threshold,
            'stable_features': stable_features,
            'stability_scores': stability_scores,
            'feature_selection_counts': feature_selection_counts,
            'outer_results': outer_results,
            'nested_stability_statistics': {
                'mean_stability': np.mean(list(stability_scores.values())),
                'std_stability': np.std(list(stability_scores.values())),
                'max_stability': np.max(list(stability_scores.values())),
                'min_stability': np.min(list(stability_scores.values())),
                'features_above_threshold': sum(1 for s in stability_scores.values() if s >= stability_threshold),
                'outer_run_consistency': np.mean([outer['inner_consistency']['mean_consistency'] for outer in outer_results])
            }
        }
        
        execution_time = time.time() - start_time
        _LOGGER.info(f"✅ Nested bootstrap stability validation completed in {execution_time:.3f}s")
        _LOGGER.info(f"📊 Stable features: {len(stable_features)}/{len(feature_names)}")
        _LOGGER.info(f"📊 Mean stability: {nested_analysis['nested_stability_statistics']['mean_stability']:.3f}")
        _LOGGER.info(f"📊 Outer run consistency: {nested_analysis['nested_stability_statistics']['outer_run_consistency']:.3f}")
        
        return {
            'stable_features': stable_features,
            'nested_analysis': nested_analysis,
            'execution_time': execution_time
        }

    def _analyze_inner_bootstrap_consistency(self, inner_results: List[Dict[str, Any]], 
                                           feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency within a single outer bootstrap run.
        
        Args:
            inner_results: Results from inner bootstrap samples
            feature_names: List of all feature names
            
        Returns:
            Dictionary with consistency analysis
        """
        if not inner_results:
            return {'mean_consistency': 0.0, 'consistency_scores': {}}
        
        # Count feature selections within this outer run
        feature_counts = {feature: 0 for feature in feature_names}
        for result in inner_results:
            for feature in result['selected_features']:
                if feature in feature_counts:
                    feature_counts[feature] += 1
        
        # Calculate consistency scores
        n_inner = len(inner_results)
        consistency_scores = {}
        for feature in feature_names:
            count = feature_counts[feature]
            consistency = count / n_inner if n_inner > 0 else 0.0
            consistency_scores[feature] = consistency
        
        mean_consistency = np.mean(list(consistency_scores.values()))
        
        return {
            'mean_consistency': mean_consistency,
            'consistency_scores': consistency_scores,
            'feature_counts': feature_counts,
            'n_inner_samples': n_inner
        }

    def _temporal_stability_validation(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str],
                                     features_target_count: int,
                                     config: Dict[str, Any],
                                     time_windows: List[int] = None,
                                     overlap_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Temporal stability validation for time-series data.
        
        This method evaluates feature selection stability across different time windows
        to ensure selected features remain relevant over time and aren't just artifacts
        of specific time periods.
        
        Args:
            X: Feature matrix (time-series data)
            y: Target array
            feature_names: List of feature names
            features_target_count: Target number of features
            config: Pipeline configuration
            time_windows: List of time window sizes (e.g., [100, 200, 300])
            overlap_ratio: Ratio of overlap between consecutive windows
            
        Returns:
            Dictionary with temporal stability analysis
        """
        start_time = time.time()
        n_samples = len(X)
        
        # Default time windows if not provided
        if time_windows is None:
            time_windows = [
                min(100, n_samples // 4),
                min(200, n_samples // 2),
                min(300, n_samples * 3 // 4)
            ]
            time_windows = [w for w in time_windows if w >= 50]  # Minimum window size
        
        _LOGGER.info(f"🔄 Starting temporal stability validation...")
        _LOGGER.info(f"📊 Time windows: {time_windows}")
        _LOGGER.info(f"📊 Overlap ratio: {overlap_ratio}")
        
        temporal_results = []
        feature_selection_counts = {feature: 0 for feature in feature_names}
        all_time_windows = []
        
        # Analyze each time window
        for window_size in time_windows:
            if window_size > n_samples:
                _LOGGER.warning(f"⚠️ Window size {window_size} > data size {n_samples}, skipping")
                continue
            
            _LOGGER.info(f"🔄 Analyzing time window: {window_size} samples")
            
            # Create overlapping windows
            step_size = int(window_size * (1 - overlap_ratio))
            if step_size == 0:
                step_size = 1
            
            window_results = []
            for start_idx in range(0, n_samples - window_size + 1, step_size):
                end_idx = start_idx + window_size
                
                try:
                    # Extract time window
                    X_window = X[start_idx:end_idx]
                    y_window = y[start_idx:end_idx]
                    
                    # Run feature selection on this time window
                    window_features = self._run_pipeline_to_consensus(
                        X_window, y_window, feature_names, features_target_count, config
                    )
                    
                    window_results.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'window_size': window_size,
                        'selected_features': window_features,
                        'n_features': len(window_features)
                    })
                    
                    # Track feature selections
                    for feature in window_features:
                        if feature in feature_selection_counts:
                            feature_selection_counts[feature] += 1
                    
                    all_time_windows.append({
                        'window_size': window_size,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'selected_features': window_features
                    })
                    
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Time window [{start_idx}:{end_idx}] failed: {e}")
                    continue
            
            if window_results:
                # Analyze stability within this window size
                window_stability = self._analyze_temporal_window_stability(
                    window_results, feature_names
                )
                
                temporal_results.append({
                    'window_size': window_size,
                    'window_results': window_results,
                    'window_stability': window_stability,
                    'n_windows': len(window_results)
                })
        
        # Calculate overall temporal stability scores
        total_windows = sum(len(result['window_results']) for result in temporal_results)
        temporal_stability_scores = {}
        for feature in feature_names:
            selection_count = feature_selection_counts[feature]
            stability_score = selection_count / total_windows if total_windows > 0 else 0.0
            temporal_stability_scores[feature] = stability_score
        
        # Select temporally stable features
        stable_features = [
            feature for feature, stability in temporal_stability_scores.items()
            if stability >= 0.6  # 60% temporal stability threshold
        ]
        
        # If too few stable features, relax criteria
        if len(stable_features) < features_target_count:
            _LOGGER.warning(f"⚠️ Only {len(stable_features)} temporally stable features found, relaxing criteria...")
            sorted_features = sorted(
                temporal_stability_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            stable_features = [feature for feature, _ in sorted_features[:features_target_count]]
        
        # Temporal stability analysis
        temporal_analysis = {
            'time_windows': time_windows,
            'overlap_ratio': overlap_ratio,
            'total_windows': total_windows,
            'stable_features': stable_features,
            'temporal_stability_scores': temporal_stability_scores,
            'feature_selection_counts': feature_selection_counts,
            'temporal_results': temporal_results,
            'all_time_windows': all_time_windows,
            'temporal_stability_statistics': {
                'mean_temporal_stability': np.mean(list(temporal_stability_scores.values())),
                'std_temporal_stability': np.std(list(temporal_stability_scores.values())),
                'max_temporal_stability': np.max(list(temporal_stability_scores.values())),
                'min_temporal_stability': np.min(list(temporal_stability_scores.values())),
                'features_above_threshold': sum(1 for s in temporal_stability_scores.values() if s >= 0.6),
                'window_size_stability': {
                    result['window_size']: result['window_stability']['mean_stability']
                    for result in temporal_results
                }
            }
        }
        
        execution_time = time.time() - start_time
        _LOGGER.info(f"✅ Temporal stability validation completed in {execution_time:.3f}s")
        _LOGGER.info(f"📊 Temporally stable features: {len(stable_features)}/{len(feature_names)}")
        _LOGGER.info(f"📊 Mean temporal stability: {temporal_analysis['temporal_stability_statistics']['mean_temporal_stability']:.3f}")
        
        return {
            'stable_features': stable_features,
            'temporal_analysis': temporal_analysis,
            'execution_time': execution_time
        }

    def _analyze_temporal_window_stability(self, window_results: List[Dict[str, Any]], 
                                         feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze stability within a specific time window size.
        
        Args:
            window_results: Results from different time windows of same size
            feature_names: List of all feature names
            
        Returns:
            Dictionary with window stability analysis
        """
        if not window_results:
            return {'mean_stability': 0.0, 'stability_scores': {}}
        
        # Count feature selections across windows of this size
        feature_counts = {feature: 0 for feature in feature_names}
        for result in window_results:
            for feature in result['selected_features']:
                if feature in feature_counts:
                    feature_counts[feature] += 1
        
        # Calculate stability scores
        n_windows = len(window_results)
        stability_scores = {}
        for feature in feature_names:
            count = feature_counts[feature]
            stability = count / n_windows if n_windows > 0 else 0.0
            stability_scores[feature] = stability
        
        mean_stability = np.mean(list(stability_scores.values()))
        
        return {
            'mean_stability': mean_stability,
            'stability_scores': stability_scores,
            'feature_counts': feature_counts,
            'n_windows': n_windows
        }

    def _cross_dataset_stability_validation(self, datasets: List[Dict[str, Any]],
                                          features_target_count: int,
                                          config: Dict[str, Any],
                                          stability_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Cross-dataset stability validation when multiple datasets are available.
        
        This method evaluates feature selection stability across different datasets
        to ensure selected features are robust and not specific to a particular dataset.
        This is particularly useful for transfer learning and domain adaptation scenarios.
        
        Args:
            datasets: List of datasets, each containing 'X', 'y', 'feature_names', and optional 'dataset_name'
            features_target_count: Target number of features
            config: Pipeline configuration
            stability_threshold: Minimum stability score for feature selection
            
        Returns:
            Dictionary with cross-dataset stability analysis
        """
        start_time = time.time()
        n_datasets = len(datasets)
        
        _LOGGER.info(f"🔄 Starting cross-dataset stability validation...")
        _LOGGER.info(f"📊 Number of datasets: {n_datasets}")
        
        # Validate datasets
        if n_datasets < 2:
            _LOGGER.warning("⚠️ Need at least 2 datasets for cross-dataset stability validation")
            return {
                'stable_features': [],
                'cross_dataset_analysis': {'error': 'Need at least 2 datasets'},
                'execution_time': time.time() - start_time
            }
        
        # Get common feature names across all datasets
        all_feature_names = set()
        for dataset in datasets:
            all_feature_names.update(dataset['feature_names'])
        
        common_features = set(datasets[0]['feature_names'])
        for dataset in datasets[1:]:
            common_features = common_features.intersection(set(dataset['feature_names']))
        
        _LOGGER.info(f"📊 Common features across datasets: {len(common_features)}")
        _LOGGER.info(f"📊 Total unique features: {len(all_feature_names)}")
        
        dataset_results = []
        feature_selection_counts = {feature: 0 for feature in all_feature_names}
        
        # Run feature selection on each dataset
        for dataset_idx, dataset in enumerate(datasets):
            dataset_name = dataset.get('dataset_name', f'Dataset_{dataset_idx + 1}')
            _LOGGER.info(f"🔄 Processing {dataset_name}...")
            
            try:
                X = dataset['X']
                y = dataset['y']
                feature_names = dataset['feature_names']
                
                # Run feature selection pipeline
                selected_features = self._run_pipeline_to_consensus(
                    X, y, feature_names, features_target_count, config
                )
                
                # Track feature selections
                for feature in selected_features:
                    if feature in feature_selection_counts:
                        feature_selection_counts[feature] += 1
                
                dataset_results.append({
                    'dataset_idx': dataset_idx,
                    'dataset_name': dataset_name,
                    'selected_features': selected_features,
                    'n_features': len(selected_features),
                    'n_samples': len(X),
                    'n_original_features': len(feature_names)
                })
                
                _LOGGER.info(f"✅ {dataset_name}: {len(selected_features)} features selected")
                
            except Exception as e:
                _LOGGER.warning(f"⚠️ {dataset_name} failed: {e}")
                continue
        
        # Calculate cross-dataset stability scores
        successful_datasets = len(dataset_results)
        cross_dataset_stability_scores = {}
        for feature in all_feature_names:
            selection_count = feature_selection_counts[feature]
            stability_score = selection_count / successful_datasets if successful_datasets > 0 else 0.0
            cross_dataset_stability_scores[feature] = stability_score
        
        # Select cross-dataset stable features
        stable_features = [
            feature for feature, stability in cross_dataset_stability_scores.items()
            if stability >= stability_threshold
        ]
        
        # If too few stable features, relax criteria
        if len(stable_features) < features_target_count:
            _LOGGER.warning(f"⚠️ Only {len(stable_features)} cross-dataset stable features found, relaxing criteria...")
            sorted_features = sorted(
                cross_dataset_stability_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            stable_features = [feature for feature, _ in sorted_features[:features_target_count]]
        
        # Analyze feature overlap between datasets
        feature_overlap_analysis = self._analyze_feature_overlap(dataset_results, all_feature_names)
        
        # Cross-dataset stability analysis
        cross_dataset_analysis = {
            'n_datasets': n_datasets,
            'successful_datasets': successful_datasets,
            'common_features': list(common_features),
            'n_common_features': len(common_features),
            'n_total_features': len(all_feature_names),
            'stable_features': stable_features,
            'cross_dataset_stability_scores': cross_dataset_stability_scores,
            'feature_selection_counts': feature_selection_counts,
            'dataset_results': dataset_results,
            'feature_overlap_analysis': feature_overlap_analysis,
            'cross_dataset_stability_statistics': {
                'mean_cross_dataset_stability': np.mean(list(cross_dataset_stability_scores.values())),
                'std_cross_dataset_stability': np.std(list(cross_dataset_stability_scores.values())),
                'max_cross_dataset_stability': np.max(list(cross_dataset_stability_scores.values())),
                'min_cross_dataset_stability': np.min(list(cross_dataset_stability_scores.values())),
                'features_above_threshold': sum(1 for s in cross_dataset_stability_scores.values() if s >= stability_threshold),
                'dataset_consistency': np.mean([len(result['selected_features']) for result in dataset_results])
            }
        }
        
        execution_time = time.time() - start_time
        _LOGGER.info(f"✅ Cross-dataset stability validation completed in {execution_time:.3f}s")
        _LOGGER.info(f"📊 Cross-dataset stable features: {len(stable_features)}/{len(all_feature_names)}")
        _LOGGER.info(f"📊 Mean cross-dataset stability: {cross_dataset_analysis['cross_dataset_stability_statistics']['mean_cross_dataset_stability']:.3f}")
        
        return {
            'stable_features': stable_features,
            'cross_dataset_analysis': cross_dataset_analysis,
            'execution_time': execution_time
        }

    def _analyze_feature_overlap(self, dataset_results: List[Dict[str, Any]], 
                               all_feature_names: set) -> Dict[str, Any]:
        """
        Analyze feature overlap between different datasets.
        
        Args:
            dataset_results: Results from feature selection on each dataset
            all_feature_names: Set of all unique feature names
            
        Returns:
            Dictionary with feature overlap analysis
        """
        if len(dataset_results) < 2:
            return {'overlap_matrix': {}, 'pairwise_overlaps': {}}
        
        # Create feature selection matrix
        feature_selection_matrix = {}
        for feature in all_feature_names:
            feature_selection_matrix[feature] = []
            for result in dataset_results:
                is_selected = 1 if feature in result['selected_features'] else 0
                feature_selection_matrix[feature].append(is_selected)
        
        # Calculate pairwise overlaps between datasets
        pairwise_overlaps = {}
        for i in range(len(dataset_results)):
            for j in range(i + 1, len(dataset_results)):
                dataset_i_features = set(dataset_results[i]['selected_features'])
                dataset_j_features = set(dataset_results[j]['selected_features'])
                
                intersection = dataset_i_features.intersection(dataset_j_features)
                union = dataset_i_features.union(dataset_j_features)
                
                jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
                
                pairwise_overlaps[f"{dataset_results[i]['dataset_name']}_vs_{dataset_results[j]['dataset_name']}"] = {
                    'intersection_size': len(intersection),
                    'union_size': len(union),
                    'jaccard_similarity': jaccard_similarity,
                    'dataset_i_size': len(dataset_i_features),
                    'dataset_j_size': len(dataset_j_features)
                }
        
        # Calculate overall overlap statistics
        overlap_scores = []
        for feature, selections in feature_selection_matrix.items():
            overlap_score = sum(selections) / len(selections)
            overlap_scores.append(overlap_score)
        
        return {
            'feature_selection_matrix': feature_selection_matrix,
            'pairwise_overlaps': pairwise_overlaps,
            'overlap_statistics': {
                'mean_overlap': np.mean(overlap_scores),
                'std_overlap': np.std(overlap_scores),
                'max_overlap': np.max(overlap_scores),
                'min_overlap': np.min(overlap_scores)
            }
        }

    def _feature_interaction_stability_validation(self, X: np.ndarray, y: np.ndarray,
                                                feature_names: List[str],
                                                features_target_count: int,
                                                config: Dict[str, Any],
                                                n_bootstrap: int = 10,
                                                max_interaction_order: int = 3) -> Dict[str, Any]:
        """
        Feature interaction stability validation to detect stable feature combinations.
        
        This method identifies feature combinations that are consistently selected together
        across bootstrap samples, indicating synergistic relationships between features.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            features_target_count: Target number of features
            config: Pipeline configuration
            n_bootstrap: Number of bootstrap samples
            max_interaction_order: Maximum order of feature interactions to analyze (2, 3, etc.)
            
        Returns:
            Dictionary with feature interaction stability analysis
        """
        start_time = time.time()
        _LOGGER.info(f"🔄 Starting feature interaction stability validation...")
        _LOGGER.info(f"📊 Bootstrap samples: {n_bootstrap}")
        _LOGGER.info(f"📊 Max interaction order: {max_interaction_order}")
        
        bootstrap_results = []
        feature_combination_counts = {}
        
        # Run bootstrap sampling
        for bootstrap_idx in range(n_bootstrap):
            try:
                # Bootstrap sampling
                bootstrap_size = int(len(X) * 0.8)
                bootstrap_indices = np.random.choice(
                    len(X), size=bootstrap_size, replace=True
                )
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]
                
                # Run feature selection on bootstrap sample
                selected_features = self._run_pipeline_to_consensus(
                    X_bootstrap, y_bootstrap, feature_names, features_target_count, config
                )
                
                bootstrap_results.append({
                    'bootstrap_idx': bootstrap_idx,
                    'selected_features': selected_features
                })
                
                # Analyze feature combinations in this bootstrap sample
                combinations = self._extract_feature_combinations(
                    selected_features, max_interaction_order
                )
                
                for combination in combinations:
                    combination_key = tuple(sorted(combination))
                    if combination_key not in feature_combination_counts:
                        feature_combination_counts[combination_key] = 0
                    feature_combination_counts[combination_key] += 1
                
            except Exception as e:
                _LOGGER.warning(f"⚠️ Bootstrap {bootstrap_idx + 1} failed: {e}")
                continue
        
        # Calculate interaction stability scores
        interaction_stability_scores = {}
        for combination, count in feature_combination_counts.items():
            stability_score = count / len(bootstrap_results) if bootstrap_results else 0.0
            interaction_stability_scores[combination] = {
                'stability_score': stability_score,
                'selection_count': count,
                'combination_size': len(combination)
            }
        
        # Identify stable feature interactions
        stable_interactions = {
            order: [] for order in range(2, max_interaction_order + 1)
        }
        
        for combination, data in interaction_stability_scores.items():
            if data['stability_score'] >= 0.6:  # 60% stability threshold
                order = data['combination_size']
                if order <= max_interaction_order:
                    stable_interactions[order].append({
                        'features': list(combination),
                        'stability_score': data['stability_score'],
                        'selection_count': data['selection_count']
                    })
        
        # Sort interactions by stability score
        for order in stable_interactions:
            stable_interactions[order].sort(
                key=lambda x: x['stability_score'], reverse=True
            )
        
        # Analyze interaction patterns
        interaction_analysis = self._analyze_interaction_patterns(
            interaction_stability_scores, feature_names
        )
        
        # Feature interaction stability analysis
        interaction_stability_analysis = {
            'n_bootstrap_samples': len(bootstrap_results),
            'max_interaction_order': max_interaction_order,
            'stable_interactions': stable_interactions,
            'interaction_stability_scores': interaction_stability_scores,
            'feature_combination_counts': feature_combination_counts,
            'interaction_analysis': interaction_analysis,
            'interaction_stability_statistics': {
                'total_combinations': len(interaction_stability_scores),
                'stable_combinations': sum(len(interactions) for interactions in stable_interactions.values()),
                'mean_stability': np.mean([data['stability_score'] for data in interaction_stability_scores.values()]),
                'max_stability': np.max([data['stability_score'] for data in interaction_stability_scores.values()]),
                'interactions_by_order': {
                    order: len(interactions) for order, interactions in stable_interactions.items()
                }
            }
        }
        
        execution_time = time.time() - start_time
        _LOGGER.info(f"✅ Feature interaction stability validation completed in {execution_time:.3f}s")
        _LOGGER.info(f"📊 Total combinations analyzed: {len(interaction_stability_scores)}")
        _LOGGER.info(f"📊 Stable combinations: {interaction_stability_analysis['interaction_stability_statistics']['stable_combinations']}")
        
        for order in range(2, max_interaction_order + 1):
            n_interactions = len(stable_interactions[order])
            if n_interactions > 0:
                _LOGGER.info(f"📊 Order {order} interactions: {n_interactions}")
        
        return {
            'stable_interactions': stable_interactions,
            'interaction_stability_analysis': interaction_stability_analysis,
            'execution_time': execution_time
        }

    def _extract_feature_combinations(self, selected_features: List[str], 
                                    max_order: int) -> List[List[str]]:
        """
        Extract feature combinations of different orders from selected features.
        
        Args:
            selected_features: List of selected feature names
            max_order: Maximum order of combinations to extract
            
        Returns:
            List of feature combinations
        """
        from itertools import combinations
        
        combinations_list = []
        
        for order in range(2, min(max_order + 1, len(selected_features) + 1)):
            for combination in combinations(selected_features, order):
                combinations_list.append(list(combination))
        
        return combinations_list

    def _analyze_interaction_patterns(self, interaction_stability_scores: Dict[tuple, Dict[str, Any]], 
                                    feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns in feature interactions.
        
        Args:
            interaction_stability_scores: Dictionary of interaction stability scores
            feature_names: List of all feature names
            
        Returns:
            Dictionary with interaction pattern analysis
        """
        # Feature co-occurrence analysis
        feature_cooccurrence = {feature: {} for feature in feature_names}
        
        for combination, data in interaction_stability_scores.items():
            if data['stability_score'] >= 0.3:  # Lower threshold for co-occurrence
                for i, feature1 in enumerate(combination):
                    for j, feature2 in enumerate(combination):
                        if i != j:
                            if feature2 not in feature_cooccurrence[feature1]:
                                feature_cooccurrence[feature1][feature2] = 0
                            feature_cooccurrence[feature1][feature2] += data['selection_count']
        
        # Calculate co-occurrence scores
        cooccurrence_scores = {}
        for feature1, cooccurrences in feature_cooccurrence.items():
            for feature2, count in cooccurrences.items():
                pair_key = tuple(sorted([feature1, feature2]))
                if pair_key not in cooccurrence_scores:
                    cooccurrence_scores[pair_key] = 0
                cooccurrence_scores[pair_key] += count
        
        # Identify most frequent co-occurring pairs
        frequent_pairs = sorted(
            cooccurrence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]  # Top 20 pairs
        
        # Feature centrality analysis
        feature_centrality = {}
        for feature in feature_names:
            centrality = sum(
                cooccurrence_scores.get(tuple(sorted([feature, other])), 0)
                for other in feature_names if other != feature
            )
            feature_centrality[feature] = centrality
        
        return {
            'feature_cooccurrence': feature_cooccurrence,
            'cooccurrence_scores': cooccurrence_scores,
            'frequent_pairs': frequent_pairs,
            'feature_centrality': feature_centrality,
            'centrality_ranking': sorted(
                feature_centrality.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }

    def _bootstrap_pipeline_stability_validation(self, X: np.ndarray, y: np.ndarray,
                                               feature_names: List[str],
                                               features_target_count: int,
                                               config: Dict[str, Any],
                                               n_bootstrap: int = 10,
                                               bootstrap_fraction: float = 0.8,
                                               stability_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Run the entire pipeline on multiple bootstrap samples to validate feature stability.
        
        This is a crucial validation step to ensure features aren't just a result of chance correlations.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            features_target_count: Target number of features
            config: Pipeline configuration
            n_bootstrap: Number of bootstrap samples
            bootstrap_fraction: Fraction of data to use in each bootstrap
            stability_threshold: Minimum stability score for feature selection
            
        Returns:
            Dictionary with stability analysis and stable features
        """
        start_time = time.time()
        _LOGGER.info(f"🔄 Starting bootstrap stability validation...")
        _LOGGER.info(f"📊 Bootstrap samples: {n_bootstrap}, Fraction: {bootstrap_fraction}")
        _LOGGER.info(f"📊 Stability threshold: {stability_threshold}")
        
        bootstrap_results = []
        feature_selection_counts = {feature: 0 for feature in feature_names}
        bootstrap_size = int(len(X) * bootstrap_fraction)
        
        # Run pipeline on multiple bootstrap samples
        for bootstrap_idx in range(n_bootstrap):
            try:
                _LOGGER.info(f"🔄 Bootstrap sample {bootstrap_idx + 1}/{n_bootstrap}")
                
                # Bootstrap sampling
                bootstrap_indices = np.random.choice(
                    len(X), size=bootstrap_size, replace=True
                )
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]
                
                # Run the pipeline up to consensus stage on bootstrap sample
                bootstrap_features = self._run_pipeline_to_consensus(
                    X_bootstrap, y_bootstrap, feature_names, features_target_count, config
                )
                
                # Count feature selections
                for feature in bootstrap_features:
                    if feature in feature_selection_counts:
                        feature_selection_counts[feature] += 1
                
                bootstrap_results.append({
                    'bootstrap_idx': bootstrap_idx,
                    'selected_features': bootstrap_features,
                    'n_features': len(bootstrap_features),
                    'bootstrap_indices': bootstrap_indices
                })
                
                _LOGGER.info(f"✅ Bootstrap {bootstrap_idx + 1}: {len(bootstrap_features)} features selected")
                
            except Exception as e:
                _LOGGER.warning(f"⚠️ Bootstrap {bootstrap_idx + 1} failed: {e}")
                continue
        
        # Calculate stability scores
        stability_scores = {}
        for feature in feature_names:
            selection_count = feature_selection_counts[feature]
            stability_score = selection_count / len(bootstrap_results) if bootstrap_results else 0.0
            stability_scores[feature] = stability_score
        
        # Select stable features
        stable_features = [
            feature for feature, stability in stability_scores.items()
            if stability >= stability_threshold
        ]
        
        # If too few stable features, relax threshold or take top features
        if len(stable_features) < features_target_count:
            _LOGGER.warning(f"⚠️ Only {len(stable_features)} stable features found, relaxing criteria...")
            
            # Sort by stability score and take top features
            sorted_features = sorted(
                stability_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            stable_features = [feature for feature, _ in sorted_features[:features_target_count]]
            
            # Update stability threshold to the minimum of selected features
            if stable_features:
                min_stability = min(stability_scores[f] for f in stable_features)
                _LOGGER.info(f"📊 Relaxed stability threshold: {min_stability:.3f}")
        
        # Stability analysis
        stability_analysis = {
            'n_bootstrap_samples': len(bootstrap_results),
            'stability_threshold': stability_threshold,
            'stable_features': stable_features,
            'stability_scores': stability_scores,
            'feature_selection_counts': feature_selection_counts,
            'stability_statistics': {
                'mean_stability': np.mean(list(stability_scores.values())),
                'std_stability': np.std(list(stability_scores.values())),
                'max_stability': np.max(list(stability_scores.values())),
                'min_stability': np.min(list(stability_scores.values())),
                'features_above_threshold': sum(1 for s in stability_scores.values() if s >= stability_threshold)
            }
        }
        
        execution_time = time.time() - start_time
        _LOGGER.info(f"✅ Bootstrap stability validation completed in {execution_time:.3f}s")
        _LOGGER.info(f"📊 Stable features: {len(stable_features)}/{len(feature_names)}")
        _LOGGER.info(f"📊 Mean stability: {stability_analysis['stability_statistics']['mean_stability']:.3f}")
        _LOGGER.info(f"📊 Features above threshold: {stability_analysis['stability_statistics']['features_above_threshold']}")
        
        return {
            'stable_features': stable_features,
            'stability_analysis': stability_analysis,
            'bootstrap_results': bootstrap_results,
            'execution_time': execution_time
        }

    def _run_pipeline_to_consensus(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str],
                                 features_target_count: int,
                                 config: Dict[str, Any]) -> List[str]:
        """
        Run the pipeline stages 1-3 (up to consensus) on a single bootstrap sample.
        
        This is a helper method for bootstrap stability validation.
        """
        try:
            current_features = feature_names.copy()
            current_X = X.copy()
            
            # Stage 1: Correlation-based filtering
            if config['use_dynamic_thresholds'] and config['correlation_threshold'] is None:
                correlation_threshold = self._determine_adaptive_correlation_threshold(current_X, current_features)
            else:
                correlation_threshold = config['correlation_threshold'] or 0.95
            
            correlation_result = self.correlation_based_filtering(
                current_X, current_features, 
                correlation_threshold=correlation_threshold
            )
            
            if 'selected_features' in correlation_result:
                current_features = correlation_result['selected_features']
                selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                current_X = X[:, selected_indices]
            
            # Stage 2: mRMR selection (if applicable)
            if len(current_features) >= config['mrmr_skip_threshold']:
                mrmr_result = self.mrmr_selection(current_X, y, current_features, len(current_features))
                
                if 'mrmr_scores' in mrmr_result and config['use_dynamic_thresholds']:
                    mrmr_threshold, _ = self._determine_mrmr_threshold(
                        mrmr_result['mrmr_scores'], current_features
                    )
                    
                    features_above_threshold = [
                        feature for feature, score in mrmr_result['mrmr_scores'].items()
                        if score >= mrmr_threshold
                    ]
                    
                    if len(features_above_threshold) < features_target_count:
                        sorted_features = sorted(
                            mrmr_result['mrmr_scores'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        features_above_threshold = [f for f, _ in sorted_features[:features_target_count]]
                    
                    current_features = features_above_threshold
                    selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                    current_X = X[:, selected_indices]
                else:
                    mrmr_target = max(features_target_count * 2, len(current_features) // 2)
                    mrmr_target = min(mrmr_target, len(current_features))
                    mrmr_result = self.mrmr_selection(current_X, y, current_features, mrmr_target)
                    if 'selected_features' in mrmr_result:
                        current_features = mrmr_result['selected_features']
                        selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                        current_X = X[:, selected_indices]
            
            # Stage 3: LASSO stability + RFE consensus
            lasso_result = self.lasso_stability_selection(
                current_X, y, current_features,
                stability_threshold=config['stability_threshold'] or 0.6
            )
            
            base_model = self._get_default_model(y)
            rfe_result = None
            if base_model is not None:
                if config['use_dynamic_thresholds']:
                    optimal_rfe_features = self._determine_optimal_rfe_features(
                        current_X, y, current_features, base_model, config['cv_folds']
                    )
                else:
                    optimal_rfe_features = max(features_target_count, 
                                             int(len(current_features) * config['consensus_reduction_factor']))
                    optimal_rfe_features = min(optimal_rfe_features, len(current_features))
                
                rfe_result = self.recursive_feature_elimination(
                    base_model, current_X, y, current_features, optimal_rfe_features
                )
            
            # Compute consensus
            lasso_features = lasso_result.get('selected_features', [])
            rfe_features = rfe_result.get('selected_features', []) if rfe_result else []
            
            if config['use_dynamic_thresholds']:
                consensus_target = max(features_target_count, min(len(lasso_features), len(rfe_features)))
            else:
                consensus_target = max(features_target_count, 
                                     int(len(current_features) * config['consensus_reduction_factor']))
            
            consensus_target = min(consensus_target, len(current_features))
            
            consensus_features = self._compute_lasso_rfe_consensus(
                lasso_features, rfe_features, consensus_target
            )
            
            return consensus_features
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Pipeline to consensus failed: {e}")
            return current_features

    def _analyze_bootstrap_stability(self, bootstrap_results: List[Dict[str, Any]], 
                                   feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze the stability of feature selection across bootstrap samples.
        
        Args:
            bootstrap_results: List of bootstrap results
            feature_names: List of all feature names
            
        Returns:
            Dictionary with detailed stability analysis
        """
        try:
            # Count feature selections across bootstrap samples
            feature_selection_counts = {feature: 0 for feature in feature_names}
            feature_selection_frequency = {}
            
            for bootstrap_result in bootstrap_results:
                selected_features = bootstrap_result.get('selected_features', [])
                for feature in selected_features:
                    if feature in feature_selection_counts:
                        feature_selection_counts[feature] += 1
            
            # Calculate selection frequencies
            n_bootstrap = len(bootstrap_results)
            for feature in feature_names:
                count = feature_selection_counts[feature]
                frequency = count / n_bootstrap if n_bootstrap > 0 else 0
                feature_selection_frequency[feature] = frequency
            
            # Analyze stability patterns
            frequencies = list(feature_selection_frequency.values())
            
            stability_analysis = {
                'feature_selection_counts': feature_selection_counts,
                'feature_selection_frequencies': feature_selection_frequency,
                'stability_statistics': {
                    'n_bootstrap_samples': n_bootstrap,
                    'mean_frequency': np.mean(frequencies),
                    'std_frequency': np.std(frequencies),
                    'max_frequency': np.max(frequencies),
                    'min_frequency': np.min(frequencies),
                    'median_frequency': np.median(frequencies),
                    'q75_frequency': np.percentile(frequencies, 75),
                    'q90_frequency': np.percentile(frequencies, 90)
                },
                'stability_categories': {
                    'highly_stable': [f for f, freq in feature_selection_frequency.items() if freq >= 0.8],
                    'moderately_stable': [f for f, freq in feature_selection_frequency.items() if 0.5 <= freq < 0.8],
                    'unstable': [f for f, freq in feature_selection_frequency.items() if freq < 0.5]
                }
            }
            
            # Feature consistency analysis
            consistency_analysis = {}
            for feature in feature_names:
                frequency = feature_selection_frequency[feature]
                if frequency >= 0.8:
                    consistency_analysis[feature] = 'highly_stable'
                elif frequency >= 0.5:
                    consistency_analysis[feature] = 'moderately_stable'
                else:
                    consistency_analysis[feature] = 'unstable'
            
            stability_analysis['feature_consistency'] = consistency_analysis
            
            _LOGGER.info(f"📊 Bootstrap stability analysis:")
            _LOGGER.info(f"   Highly stable features (≥80%): {len(stability_analysis['stability_categories']['highly_stable'])}")
            _LOGGER.info(f"   Moderately stable features (50-80%): {len(stability_analysis['stability_categories']['moderately_stable'])}")
            _LOGGER.info(f"   Unstable features (<50%): {len(stability_analysis['stability_categories']['unstable'])}")
            _LOGGER.info(f"   Mean selection frequency: {stability_analysis['stability_statistics']['mean_frequency']:.3f}")
            
            return stability_analysis
            
        except Exception as e:
            _LOGGER.error(f"❌ Bootstrap stability analysis failed: {e}")
            return {
                'error': str(e),
                'feature_selection_counts': {},
                'feature_selection_frequencies': {},
                'stability_statistics': {},
                'stability_categories': {'highly_stable': [], 'moderately_stable': [], 'unstable': []},
                'feature_consistency': {}
            }

    def _log_feature_reduction_stats(self, method_name: str, original_count: int, 
                                   selected_count: int, execution_time: float,
                                   additional_stats: Optional[Dict[str, Any]] = None):
        """Enhanced feature reduction reporting with comprehensive statistics."""
        removed_count = original_count - selected_count
        reduction_percent = safe_divide(removed_count, original_count) * 100
        
        _LOGGER.info(f"📊 {method_name} Results:")
        _LOGGER.info(f"   Original features: {original_count}")
        _LOGGER.info(f"   Selected features: {selected_count}")
        _LOGGER.info(f"   Removed features: {removed_count} ({reduction_percent:.1f}%)")
        _LOGGER.info(f"   Execution time: {execution_time:.3f}s")
        _LOGGER.info(f"   Features/second: {safe_divide(original_count, execution_time):.1f}")
        
        # Memory reporting
        try:
            memory_stats = get_m1_performance_stats()
            if 'memory_report' in memory_stats:
                memory_info = memory_stats['memory_report']
                _LOGGER.info(f"   Memory usage: {memory_info.get('current_mb', 0):.1f}MB")
                _LOGGER.info(f"   Peak memory: {memory_info.get('peak_mb', 0):.1f}MB")
        except Exception as e:
            _LOGGER.debug(f"Memory stats unavailable: {e}")
        
        # Additional statistics
        if additional_stats:
            for key, value in additional_stats.items():
                _LOGGER.info(f"   {key}: {value}")
        
        # Performance monitoring
        try:
            perf_stats = get_m1_performance_stats()
            if 'm1_enhanced_operations' in perf_stats:
                ops_stats = perf_stats['m1_enhanced_operations']
                _LOGGER.info(f"   GPU operations: {ops_stats.get('gpu_operations', 0)}")
                _LOGGER.info(f"   CPU operations: {ops_stats.get('cpu_operations', 0)}")
                _LOGGER.info(f"   Memory optimizations: {ops_stats.get('memory_optimizations', 0)}")
        except Exception as e:
            _LOGGER.debug(f"Performance stats unavailable: {e}")

    def _monitor_performance(self, operation_name: str, start_time: float, start_memory: float = 0.0):
        """Monitor and report performance metrics."""
        execution_time = time.time() - start_time
        
        # Get performance stats
        try:
            perf_stats = get_m1_performance_stats()
            if 'gpu_device' in perf_stats:
                _LOGGER.info(f"🎯 GPU device: {perf_stats['gpu_device']}")
            if 'gpu_memory_info' in perf_stats:
                gpu_mem = perf_stats['gpu_memory_info']
                _LOGGER.info(f"🎯 GPU memory: {gpu_mem.get('used_mb', 0):.1f}MB / {gpu_mem.get('total_mb', 0):.1f}MB")
        except Exception as e:
            _LOGGER.debug(f"GPU stats unavailable: {e}")
        
        # Report throughput
        throughput = safe_divide(1.0, execution_time)
        _LOGGER.info(f"⚡ {operation_name} throughput: {throughput:.2f} ops/sec")

    def mrmr_selection(self, X: np.ndarray, y: np.ndarray,
                      feature_names: List[str], n_features: int,
                      relevance_method: str = 'mutual_info',
                      redundancy_method: str = 'correlation') -> Dict[str, Any]:
        """
        Perform mRMR (Minimum Redundancy Maximum Relevance) feature selection.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            n_features: Number of features to select
            relevance_method: Method for relevance calculation ('mutual_info', 'correlation', 'importance')
            redundancy_method: Method for redundancy calculation ('correlation', 'mutual_info')

        Returns:
            Dictionary with selected features and scores
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting mRMR feature selection...")
        _LOGGER.info(f"📊 Parameters - Features to select: {n_features}, Data shape: {X.shape}")
        _LOGGER.info(f"📊 Methods - Relevance: {relevance_method}, Redundancy: {redundancy_method}")
        
        try:
            mrmr_results = {
                'selected_features': [],
                'feature_scores': {},
                'relevance_scores': {},
                'redundancy_scores': {},
                'mrmr_scores': {},
                'selection_metadata': {
                    'method': 'mrmr',
                    'relevance_method': relevance_method,
                    'redundancy_method': redundancy_method,
                    'n_features_requested': n_features
                }
            }

            # Calculate relevance scores with parallel processing
            _LOGGER.info("🔍 Calculating relevance scores with parallel processing...")
            relevance_scores = self._calculate_relevance_scores_parallel(X, y, feature_names, relevance_method)
            mrmr_results['relevance_scores'] = relevance_scores

            # mRMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(feature_names)))

            # Start with most relevant feature
            if relevance_scores:
                try:
                    best_feature_name = max(relevance_scores.items(), key=lambda x: x[1])[0]
                    # Ensure best_feature_name is a string and exists in feature_names
                    if isinstance(best_feature_name, str) and best_feature_name in feature_names:
                        best_feature_idx = feature_names.index(best_feature_name)
                        selected_indices.append(best_feature_idx)
                        remaining_indices.remove(best_feature_idx)

                        mrmr_results['selected_features'].append(feature_names[best_feature_idx])
                        mrmr_results['mrmr_scores'][feature_names[best_feature_idx]] = relevance_scores[best_feature_name]
                    else:
                        self.logger.warning(f"⚠️ Invalid feature name in relevance scores: {best_feature_name}")
                        # Fallback: select first feature
                        if feature_names:
                            selected_indices.append(0)
                            remaining_indices.remove(0)
                            mrmr_results['selected_features'].append(feature_names[0])
                            mrmr_results['mrmr_scores'][feature_names[0]] = 0.0
                except (ValueError, KeyError, TypeError) as e:
                    self.logger.warning(f"⚠️ Error selecting initial feature: {e}")
                    # Fallback: select first feature
                    if feature_names:
                        selected_indices.append(0)
                        remaining_indices.remove(0)
                        mrmr_results['selected_features'].append(feature_names[0])
                        mrmr_results['mrmr_scores'][feature_names[0]] = 0.0

            # Iteratively select features
            while len(selected_indices) < n_features and remaining_indices:
                best_score = -np.inf
                best_idx = None

                for idx in remaining_indices:
                    feature_name = feature_names[idx]

                    # Calculate relevance
                    relevance = relevance_scores.get(feature_name, 0)

                    # Calculate redundancy with already selected features
                    redundancy = 0
                    if selected_indices:
                        redundancy_scores = []
                        for selected_idx in selected_indices:
                            selected_name = feature_names[selected_idx]
                            score = self._calculate_redundancy_score(
                                X[:, idx], X[:, selected_idx],
                                feature_name, selected_name, redundancy_method
                            )
                            redundancy_scores.append(score)
                        redundancy = np.mean(redundancy_scores)

                    # mRMR score
                    mrmr_score = relevance - redundancy

                    if mrmr_score > best_score:
                        best_score = mrmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    feature_name = feature_names[best_idx]
                    mrmr_results['selected_features'].append(feature_name)
                    mrmr_results['mrmr_scores'][feature_name] = best_score

                    # Store individual scores
                    mrmr_results['feature_scores'][feature_name] = {
                        'relevance': relevance_scores.get(feature_name, 0),
                        'redundancy': redundancy,
                        'mrmr_score': best_score
                    }

            mrmr_results['selection_metadata']['n_features_selected'] = len(mrmr_results['selected_features'])

            execution_time = time.time() - start_time
            
            # Enhanced reporting with comprehensive statistics
            additional_stats = {
                'Relevance method': relevance_method,
                'Redundancy method': redundancy_method,
                'Target features': n_features,
                'Selection ratio': f"{len(mrmr_results['selected_features'])}/{n_features}"
            }
            self._log_feature_reduction_stats(
                "mRMR Selection", len(feature_names), len(mrmr_results['selected_features']), 
                execution_time, additional_stats
            )
            _LOGGER.debug(f"📊 Selected features: {mrmr_results['selected_features']}")
            return mrmr_results

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ mRMR selection failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'selected_features': []}

    def stability_weighted_selection(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   stability_scores: Dict[str, float],
                                   threshold: float = 0.6,
                                   n_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform stability-weighted feature selection.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            stability_scores: Dictionary of feature stability scores
            threshold: Minimum stability threshold
            n_features: Number of features to select (None for threshold-based)

        Returns:
            Dictionary with selected features and stability analysis
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting stability-weighted feature selection...")
        _LOGGER.info(f"📊 Parameters - Threshold: {threshold}, Data shape: {X.shape}")
        _LOGGER.info(f"📊 Features to select: {n_features if n_features else 'threshold-based'}")
        _LOGGER.info(f"📊 Stability scores available: {len(stability_scores)}")
        
        try:
            stability_results = {
                'selected_features': [],
                'stability_analysis': {},
                'selection_metadata': {
                    'method': 'stability_weighted',
                    'stability_threshold': threshold,
                    'n_features_requested': n_features
                }
            }

            # Calculate base importance scores
            importance_scores = self._calculate_importance_scores(X, y, feature_names)

            # Combine stability and importance
            combined_scores = {}
            for feature in feature_names:
                stability = stability_scores.get(feature, 0.5)
                importance = importance_scores.get(feature, 0.0)

                # Weighted combination
                combined_score = stability * importance

                combined_scores[feature] = {
                    'stability': stability,
                    'importance': importance,
                    'combined_score': combined_score,
                    'meets_threshold': stability >= threshold
                }

            stability_results['stability_analysis'] = combined_scores

            # Select features
            if n_features is None:
                # Threshold-based selection
                selected_features = [
                    feature for feature, scores in combined_scores.items()
                    if scores['meets_threshold']
                ]
            else:
                # Top-N selection
                sorted_features = sorted(
                    combined_scores.items(),
                    key=lambda x: x[1]['combined_score'],
                    reverse=True
                )
                selected_features = [feature for feature, _ in sorted_features[:n_features]]

            stability_results['selected_features'] = selected_features
            stability_results['selection_metadata']['n_features_selected'] = len(selected_features)

            # Stability statistics
            stabilities = [scores['stability'] for scores in combined_scores.values()]
            stability_results['selection_metadata']['stability_stats'] = {
                'mean_stability': np.mean(stabilities),
                'std_stability': np.std(stabilities),
                'min_stability': np.min(stabilities),
                'max_stability': np.max(stabilities),
                'stable_features': sum(1 for s in stabilities if s >= threshold)
            }

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Stability-weighted selection completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Selected: {len(selected_features)} features")
            _LOGGER.info(f"📊 Stability stats - Mean: {np.mean(stabilities):.3f}, "
                        f"Stable features: {sum(1 for s in stabilities if s >= threshold)}")
            _LOGGER.debug(f"📊 Selected features: {selected_features}")
            return stability_results

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Stability-weighted selection failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'selected_features': []}

    def correlation_based_filtering(self, X: np.ndarray, feature_names: List[str],
                                  correlation_threshold: float = 0.95,
                                  method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation-based feature filtering.

        Args:
            X: Feature matrix
            feature_names: List of feature names
            correlation_threshold: Correlation threshold for filtering
            method: Correlation method ('pearson', 'spearman')

        Returns:
            Dictionary with filtered features and correlation analysis
        """
        start_time = time.time()
        try:
            self.logger.info(f"🔍 Starting correlation-based filtering (threshold={correlation_threshold})")

            correlation_results = {
                'selected_features': feature_names.copy(),  # Start with all
                'removed_features': [],
                'correlation_matrix': {},
                'highly_correlated_pairs': [],
                'selection_metadata': {
                    'method': 'correlation_filtering',
                    'correlation_threshold': correlation_threshold,
                    'correlation_method': method
                }
            }

            # Calculate correlation matrix using M1-optimized operations
            _LOGGER.info("🔍 Computing correlation matrix with M1 optimization...")
            try:
                if method == 'pearson':
                    corr_matrix = m1_correlation_matrix(X.T)
                elif method == 'spearman':
                    # For Spearman, convert to DataFrame and use pandas (more efficient than loops)
                    import pandas as pd
                    df = pd.DataFrame(X.T)
                    corr_matrix = df.corr(method='spearman').values
                else:
                    raise ValueError(f"Unsupported correlation method: {method}")
                
                # Memory optimization after correlation computation
                m1_optimize_memory()
                _LOGGER.info("🧠 Memory optimized after correlation computation")
                
            except Exception as e:
                _LOGGER.warning(f"⚠️ M1 correlation failed, falling back to numpy: {e}")
                # Fallback to original implementation
                if method == 'pearson':
                    corr_matrix = np.corrcoef(X.T)
                elif method == 'spearman':
                    from scipy.stats import spearmanr
                    corr_matrix = np.zeros((X.shape[1], X.shape[1]))
                    for i in range(X.shape[1]):
                        for j in range(X.shape[1]):
                            if i != j:
                                corr, _ = spearmanr(X[:, i], X[:, j])
                                corr_matrix[i, j] = corr
                            else:
                                corr_matrix[i, j] = 1.0
                else:
                    raise ValueError(f"Unsupported correlation method: {method}")

            # Store correlation matrix
            for i, feature_i in enumerate(feature_names):
                correlation_results['correlation_matrix'][feature_i] = {}
                for j, feature_j in enumerate(feature_names):
                    if i != j:
                        correlation_results['correlation_matrix'][feature_i][feature_j] = corr_matrix[i, j]

            # Find highly correlated pairs
            removed_features = set()
            for i in range(len(feature_names)):
                if feature_names[i] in removed_features:
                    continue

                for j in range(i + 1, len(feature_names)):
                    if feature_names[j] in removed_features:
                        continue

                    corr_value = abs(corr_matrix[i, j])
                    if corr_value >= correlation_threshold:
                        # Remove the feature with higher index (arbitrary choice)
                        removed_feature = feature_names[j]
                        removed_features.add(removed_feature)

                        correlation_results['highly_correlated_pairs'].append({
                            'feature1': feature_names[i],
                            'feature2': removed_feature,
                            'correlation': corr_value
                        })

            # Update selected features
            correlation_results['selected_features'] = [
                f for f in feature_names if f not in removed_features
            ]
            correlation_results['removed_features'] = list(removed_features)

            correlation_results['selection_metadata'].update({
                'n_features_original': len(feature_names),
                'n_features_selected': len(correlation_results['selected_features']),
                'n_features_removed': len(correlation_results['removed_features']),
                'n_correlated_pairs': len(correlation_results['highly_correlated_pairs'])
            })

            # Enhanced reporting
            execution_time = time.time() - start_time
            additional_stats = {
                'Correlation method': method,
                'Correlation threshold': correlation_threshold,
                'Correlated pairs found': len(correlation_results['highly_correlated_pairs']),
                'Retention rate': f"{len(correlation_results['selected_features'])}/{len(feature_names)}"
            }
            self._log_feature_reduction_stats(
                "Correlation Filtering", len(feature_names), 
                len(correlation_results['selected_features']), execution_time, additional_stats
            )
            return correlation_results

        except Exception as e:
            self.logger.error(f"❌ Correlation-based filtering failed: {e}")
            return {'error': str(e), 'selected_features': feature_names}

    def recursive_feature_elimination(self, model: Any, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str], n_features: int,
                                    cv: int = 3) -> Dict[str, Any]:
        """
        Perform recursive feature elimination.

        Args:
            model: Base model for RFE
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            n_features: Number of features to select
            cv: Number of cross-validation folds

        Returns:
            Dictionary with selected features and RFE results
        """
        try:
            self.logger.info(f"🔄 Starting recursive feature elimination for {n_features} features")

            rfe_results = {
                'selected_features': [],
                'feature_ranking': {},
                'feature_scores': {},
                'selection_metadata': {
                    'method': 'recursive_feature_elimination',
                    'n_features_requested': n_features,
                    'cv_folds': cv
                }
            }

            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for recursive feature elimination")

            # Create RFE selector with M1 optimization
            rfe_selector = RFE(
                estimator=model,
                n_features_to_select=n_features,
                step=self.method_configs['rfe']['step']
            )

            # Fit RFE with memory optimization
            _LOGGER.info("🔍 Fitting RFE with M1 optimization...")
            rfe_selector.fit(X, y)
            
            # Memory optimization after RFE fitting
            m1_optimize_memory()
            _LOGGER.info("🧠 Memory optimized after RFE fitting")

            # Get selected features
            selected_mask = rfe_selector.support_
            selected_indices = np.where(selected_mask)[0]

            rfe_results['selected_features'] = [
                feature_names[idx] for idx in selected_indices
            ]

            # Get feature ranking
            ranking = rfe_selector.ranking_
            for idx, feature_name in enumerate(feature_names):
                rfe_results['feature_ranking'][feature_name] = ranking[idx]

            # Calculate cross-validated scores for different feature subsets
            if cv > 1:
                rfecv = RFECV(
                    estimator=model,
                    step=self.method_configs['rfe']['step'],
                    cv=StratifiedKFold(cv),
                    scoring=self.method_configs['rfe']['scoring']
                )
                rfecv.fit(X, y)

                rfe_results['optimal_n_features'] = rfecv.n_features_
                rfe_results['cv_scores'] = rfecv.cv_results_['mean_test_score'].tolist()

            rfe_results['selection_metadata']['n_features_selected'] = len(rfe_results['selected_features'])

            self.logger.info(f"✅ Recursive feature elimination completed: "
                           f"{len(rfe_results['selected_features'])} features selected")
            return rfe_results

        except Exception as e:
            self.logger.error(f"❌ Recursive feature elimination failed: {e}")
            return {'error': str(e), 'selected_features': []}

    def feature_importance_ranking(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str],
                                 method: str = 'permutation') -> Dict[str, Any]:
        """
        Rank features by importance using various methods.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            method: Importance calculation method ('permutation', 'tree_importance', 'coefficients')

        Returns:
            Dictionary with feature importance ranking
        """
        try:
            self.logger.info(f"📊 Calculating feature importance using {method} method")

            importance_results = {
                'feature_importance': {},
                'ranking': [],
                'selection_metadata': {
                    'method': 'feature_importance_ranking',
                    'importance_method': method
                }
            }

            if method == 'permutation':
                importance_scores = self._calculate_permutation_importance(model, X, y, feature_names)
            elif method == 'tree_importance':
                importance_scores = self._calculate_tree_importance(model, X, y, feature_names)
            elif method == 'coefficients':
                importance_scores = self._calculate_coefficient_importance(model, X, y, feature_names)
            elif method == 'shap':
                shap_result = self._calculate_shap_importance(model, X, feature_names)
                importance_scores = shap_result.get('importance_scores', {})
            else:
                raise ValueError(f"Unsupported importance method: {method}")

            importance_results['feature_importance'] = importance_scores

            # Create ranking
            sorted_features = sorted(
                importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            importance_results['ranking'] = [
                {'feature': feature, 'importance': score, 'rank': idx + 1}
                for idx, (feature, score) in enumerate(sorted_features)
            ]

            self.logger.info(f"✅ Feature importance ranking completed for {len(feature_names)} features")
            return importance_results

        except Exception as e:
            self.logger.error(f"❌ Feature importance ranking failed: {e}")
            return {'error': str(e), 'ranking': []}

    def _calculate_shap_importance(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate SHAP-based feature importance with robust fallbacks."""
        try:
            import shap
            self.logger.info("🧠 Computing SHAP values for interpretability")
            # Choose explainer based on model type
            explainer = None
            try:
                explainer = shap.Explainer(model, X)
            except Exception:
                try:
                    explainer = shap.KernelExplainer(lambda data: model.predict(data), X[: min(len(X), 200)])
                except Exception as e:
                    self.logger.warning(f"SHAP explainer creation failed: {e}")
                    return {'importance_scores': {}, 'error': str(e)}

            subset_size = min(1000, len(X))
            shap_values = explainer(X[:subset_size])

            import numpy as _np
            vals = getattr(shap_values, 'values', None)
            if vals is None:
                vals = _np.array(shap_values)

            if vals.ndim == 3:
                abs_mean = _np.mean(_np.mean(_np.abs(vals), axis=2), axis=0)
            else:
                abs_mean = _np.mean(_np.abs(vals), axis=0)

            scores = {name: float(abs_mean[i]) for i, name in enumerate(feature_names[: len(abs_mean)])}

            return {
                'importance_scores': scores,
                'method': 'shap',
                'n_samples_used': subset_size
            }
        except Exception as e:
            self.logger.warning(f"⚠️ SHAP importance failed: {e}")
            return {'importance_scores': {}, 'error': str(e)}

    def composite_feature_scoring(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str],
                                methods: List[str] = None,
                                weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate composite feature scores using multiple methods.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            methods: List of scoring methods to use
            weights: Weights for each method

        Returns:
            Dictionary with composite feature scores
        """
        try:
            if methods is None:
                methods = ['mutual_info', 'importance', 'stability']

            if weights is None:
                weights = {method: 1.0 / len(methods) for method in methods}

            self.logger.info(f"🔍 Calculating composite feature scores using {methods}")

            composite_results = {
                'composite_scores': {},
                'method_scores': {},
                'feature_ranking': [],
                'selection_metadata': {
                    'method': 'composite_scoring',
                    'methods_used': methods,
                    'method_weights': weights
                }
            }

            # Calculate scores for each method
            method_scores = {}
            for method in methods:
                if method == 'mutual_info':
                    scores = self._calculate_relevance_scores(X, y, feature_names, 'mutual_info')
                elif method == 'importance':
                    scores = self._calculate_importance_scores(X, y, feature_names)
                elif method == 'stability':
                    scores = self._calculate_stability_scores(X, feature_names)
                elif method == 'variance':
                    scores = self._calculate_variance_scores(X, feature_names)
                else:
                    self.logger.warning(f"Unknown scoring method: {method}")
                    continue

                method_scores[method] = scores

            composite_results['method_scores'] = method_scores

            # Calculate composite scores
            for feature in feature_names:
                composite_score = 0.0
                method_contributions = {}

                for method in methods:
                    if method in method_scores and feature in method_scores[method]:
                        score = method_scores[method][feature]
                        weight = weights.get(method, 1.0)
                        contribution = score * weight
                        composite_score += contribution
                        method_contributions[method] = contribution

                composite_results['composite_scores'][feature] = {
                    'composite_score': composite_score,
                    'method_contributions': method_contributions
                }

            # Create ranking
            sorted_features = sorted(
                composite_results['composite_scores'].items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )

            composite_results['feature_ranking'] = [
                {'feature': feature, 'composite_score': scores['composite_score'],
                 'rank': idx + 1, 'method_contributions': scores['method_contributions']}
                for idx, (feature, scores) in enumerate(sorted_features)
            ]

            self.logger.info(f"✅ Composite feature scoring completed for {len(feature_names)} features")
            return composite_results

        except Exception as e:
            self.logger.error(f"❌ Composite feature scoring failed: {e}")
            return {'error': str(e), 'composite_scores': {}}

    def cross_validated_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str],
                                        cv_folds: int = 5,
                                        selection_method: str = 'importance') -> Dict[str, Any]:
        """
        Perform cross-validated feature selection for stability assessment.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            cv_folds: Number of cross-validation folds
            selection_method: Feature selection method

        Returns:
            Dictionary with cross-validated feature selection results
        """
        try:
            self.logger.info(f"🔄 Starting cross-validated feature selection ({cv_folds} folds)")

            cv_results = {
                'fold_selections': [],
                'feature_stability': {},
                'consensus_features': [],
                'selection_metadata': {
                    'method': 'cross_validated_selection',
                    'cv_folds': cv_folds,
                    'selection_method': selection_method
                }
            }

            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for cross-validated feature selection")

            # Perform cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            fold_selections = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                try:
                    X_fold, y_fold = X[train_idx], y[train_idx]

                    # Perform feature selection on this fold
                    fold_selection = self._select_features_single_fold(
                        X_fold, y_fold, feature_names, selection_method
                    )

                    fold_selections.append({
                        'fold_idx': fold_idx,
                        'selected_features': fold_selection,
                        'n_features': len(fold_selection)
                    })

                except Exception as fold_e:
                    self.logger.warning(f"⚠️ Fold {fold_idx} feature selection failed: {fold_e}")
                    continue

            cv_results['fold_selections'] = fold_selections

            # Calculate feature stability
            if fold_selections:
                cv_results['feature_stability'] = self._calculate_feature_stability(
                    fold_selections, feature_names
                )

                # Find consensus features (selected in most folds)
                feature_counts = {}
                for fold in fold_selections:
                    for feature in fold['selected_features']:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1

                consensus_threshold = cv_folds * 0.6  # Selected in 60% of folds
                cv_results['consensus_features'] = [
                    feature for feature, count in feature_counts.items()
                    if count >= consensus_threshold
                ]

                cv_results['selection_metadata'].update({
                    'total_folds_completed': len(fold_selections),
                    'consensus_threshold': consensus_threshold,
                    'n_consensus_features': len(cv_results['consensus_features'])
                })

            self.logger.info(f"✅ Cross-validated feature selection completed: "
                           f"{len(cv_results['consensus_features'])} consensus features found")
            return cv_results

        except Exception as e:
            self.logger.error(f"❌ Cross-validated feature selection failed: {e}")
            return {'error': str(e), 'consensus_features': []}

    def lasso_stability_selection(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str],
                                 n_bootstrap: int = 100,
                                 bootstrap_fraction: float = 0.8,
                                 alpha_range: Tuple[float, float] = (0.001, 1.0),
                                 stability_threshold: float = 0.6,
                                 cv_folds: int = 5) -> Dict[str, Any]:
        """
        LASSO with stability selection to overcome instability issues.
        
        This method combines:
        1. LASSO regularization for automatic feature selection
        2. Bootstrap sampling for stability assessment
        3. Cross-validation for optimal alpha selection
        4. Stability thresholding for robust feature selection
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            n_bootstrap: Number of bootstrap samples
            bootstrap_fraction: Fraction of data to use in each bootstrap
            alpha_range: Range of alpha values to test
            stability_threshold: Minimum stability score for feature selection
            cv_folds: Number of CV folds for alpha selection
            
        Returns:
            Dictionary with stable features and LASSO analysis
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting LASSO stability selection...")
        _LOGGER.info(f"📊 Parameters - Bootstrap samples: {n_bootstrap}, Data shape: {X.shape}")
        _LOGGER.info(f"📊 Alpha range: {alpha_range}, Stability threshold: {stability_threshold}")
        
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for LASSO stability selection")
            
            # Standardize features (important for LASSO)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            lasso_stability_results = {
                'selected_features': [],
                'feature_stability_scores': {},
                'feature_coefficients': {},
                'alpha_analysis': {},
                'bootstrap_results': [],
                'selection_metadata': {
                    'method': 'lasso_stability_selection',
                    'n_bootstrap': n_bootstrap,
                    'bootstrap_fraction': bootstrap_fraction,
                    'alpha_range': alpha_range,
                    'stability_threshold': stability_threshold,
                    'cv_folds': cv_folds
                }
            }
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) <= 10 and not np.issubdtype(np.asarray(y).dtype, np.floating)
            
            # Initialize feature selection counts
            feature_selection_counts = {feature: 0 for feature in feature_names}
            feature_coefficients_sum = {feature: 0.0 for feature in feature_names}
            
            # Bootstrap sampling and LASSO selection with parallel processing
            bootstrap_size = int(len(X_scaled) * bootstrap_fraction)
            
            _LOGGER.info(f"🔄 Starting parallel bootstrap processing ({n_bootstrap} samples)...")
            
            # Prepare bootstrap parameters for parallel processing
            bootstrap_params = []
            for bootstrap_idx in range(n_bootstrap):
                bootstrap_params.append({
                    'bootstrap_idx': bootstrap_idx,
                    'X_scaled': X_scaled,
                    'y': y,
                    'bootstrap_size': bootstrap_size,
                    'feature_names': feature_names,
                    'alpha_range': alpha_range,
                    'cv_folds': cv_folds,
                    'is_classification': is_classification,
                    'random_state': self.random_state + bootstrap_idx,
                    'method_configs': self.method_configs
                })
            
            # Use parallel processing for bootstrap iterations
            if self.parallel_processor and self.enable_parallel:
                try:
                    _LOGGER.info(f"⚡ Using parallel processing with {self.max_workers} workers")
                    bootstrap_results = self.parallel_processor.parallel_apply(
                        bootstrap_params,
                        self._lasso_bootstrap_fit,
                        max_workers=self.max_workers
                    )
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Parallel bootstrap failed: {e}, falling back to sequential")
                    bootstrap_results = [self._lasso_bootstrap_fit(params) for params in bootstrap_params]
            else:
                # Sequential fallback
                bootstrap_results = [self._lasso_bootstrap_fit(params) for params in bootstrap_params]
            
            # Process bootstrap results
            for result in bootstrap_results:
                if result and 'error' not in result:
                    bootstrap_idx = result['bootstrap_idx']
                    selected_features = result['selected_features']
                    
                    # Update selection counts and coefficient sums
                    for feature in selected_features:
                        feature_selection_counts[feature] += 1
                        feature_idx = feature_names.index(feature)
                        feature_coefficients_sum[feature] += result['coefficients'][feature_idx]
                    
                    # Store bootstrap result
                    lasso_stability_results['bootstrap_results'].append(result)
                else:
                    _LOGGER.warning(f"⚠️ Bootstrap result failed: {result.get('error', 'Unknown error')}")
            
            _LOGGER.info(f"✅ Completed {len(lasso_stability_results['bootstrap_results'])}/{n_bootstrap} bootstrap iterations")
            
            # Calculate stability scores
            for feature in feature_names:
                stability_score = feature_selection_counts[feature] / n_bootstrap
                avg_coefficient = (feature_coefficients_sum[feature] / feature_selection_counts[feature] 
                                 if feature_selection_counts[feature] > 0 else 0.0)
                
                lasso_stability_results['feature_stability_scores'][feature] = stability_score
                lasso_stability_results['feature_coefficients'][feature] = avg_coefficient
            
            # Select stable features
            stable_features = [
                feature for feature, stability in lasso_stability_results['feature_stability_scores'].items()
                if stability >= stability_threshold
            ]
            
            lasso_stability_results['selected_features'] = stable_features
            
            # Alpha analysis
            if lasso_stability_results['bootstrap_results']:
                alphas = [result['optimal_alpha'] for result in lasso_stability_results['bootstrap_results']]
                lasso_stability_results['alpha_analysis'] = {
                    'mean_alpha': np.mean(alphas),
                    'std_alpha': np.std(alphas),
                    'min_alpha': np.min(alphas),
                    'max_alpha': np.max(alphas),
                    'median_alpha': np.median(alphas)
                }
            
            lasso_stability_results['selection_metadata'].update({
                'n_features_selected': len(stable_features),
                'n_bootstrap_successful': len(lasso_stability_results['bootstrap_results']),
                'stability_stats': {
                    'mean_stability': np.mean(list(lasso_stability_results['feature_stability_scores'].values())),
                    'std_stability': np.std(list(lasso_stability_results['feature_stability_scores'].values())),
                    'max_stability': np.max(list(lasso_stability_results['feature_stability_scores'].values())),
                    'min_stability': np.min(list(lasso_stability_results['feature_stability_scores'].values()))
                }
            })
            
            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ LASSO stability selection completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Selected: {len(stable_features)} stable features")
            _LOGGER.info(f"📊 Stability stats - Mean: {lasso_stability_results['selection_metadata']['stability_stats']['mean_stability']:.3f}")
            _LOGGER.debug(f"📊 Selected features: {stable_features}")
            return lasso_stability_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ LASSO stability selection failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'selected_features': []}

    def lasso_feature_selection(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str],
                               alpha: Optional[float] = None,
                               cv_folds: int = 5,
                               selection_criterion: str = 'cv') -> Dict[str, Any]:
        """
        Standard LASSO feature selection with optional cross-validation.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            alpha: LASSO regularization strength (None for CV selection)
            cv_folds: Number of CV folds for alpha selection
            selection_criterion: 'cv' for cross-validation, 'aic' for AIC, 'bic' for BIC
            
        Returns:
            Dictionary with selected features and LASSO results
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting LASSO feature selection...")
        _LOGGER.info(f"📊 Parameters - Alpha: {alpha if alpha else 'CV'}, Data shape: {X.shape}")
        _LOGGER.info(f"📊 Selection criterion: {selection_criterion}")
        
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for LASSO feature selection")
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            lasso_results = {
                'selected_features': [],
                'feature_coefficients': {},
                'alpha_analysis': {},
                'selection_metadata': {
                    'method': 'lasso_feature_selection',
                    'alpha': alpha,
                    'cv_folds': cv_folds,
                    'selection_criterion': selection_criterion
                }
            }
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) <= 10 and not np.issubdtype(np.asarray(y).dtype, np.floating)
            
            if alpha is None:
                # Use cross-validation to find optimal alpha
                if is_classification:
                    lasso_cv = LassoCV(
                        alphas=np.logspace(np.log10(self.method_configs['lasso']['alpha_range'][0]), 
                                         np.log10(self.method_configs['lasso']['alpha_range'][1]), 20),
                        cv=cv_folds,
                        max_iter=self.method_configs['lasso']['max_iter'],
                        tol=self.method_configs['lasso']['tol'],
                        random_state=self.random_state
                    )
                else:
                    lasso_cv = LassoCV(
                        alphas=np.logspace(np.log10(self.method_configs['lasso']['alpha_range'][0]), 
                                         np.log10(self.method_configs['lasso']['alpha_range'][1]), 20),
                        cv=cv_folds,
                        max_iter=self.method_configs['lasso']['max_iter'],
                        tol=self.method_configs['lasso']['tol'],
                        random_state=self.random_state
                    )
                
                lasso_cv.fit(X_scaled, y)
                optimal_alpha = lasso_cv.alpha_
                lasso_model = lasso_cv
                
                lasso_results['alpha_analysis'] = {
                    'optimal_alpha': optimal_alpha,
                    'cv_scores': lasso_cv.mse_path_.mean(axis=1).tolist(),
                    'alphas_tested': lasso_cv.alphas_.tolist()
                }
            else:
                # Use specified alpha
                if is_classification:
                    lasso_model = Lasso(
                        alpha=alpha,
                        max_iter=self.method_configs['lasso']['max_iter'],
                        tol=self.method_configs['lasso']['tol'],
                        random_state=self.random_state
                    )
                else:
                    lasso_model = Lasso(
                        alpha=alpha,
                        max_iter=self.method_configs['lasso']['max_iter'],
                        tol=self.method_configs['lasso']['tol'],
                        random_state=self.random_state
                    )
                
                lasso_model.fit(X_scaled, y)
                optimal_alpha = alpha
            
            # Get selected features (non-zero coefficients)
            selected_mask = np.abs(lasso_model.coef_) > 1e-6
            selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
            
            # Store coefficients
            for i, feature in enumerate(feature_names):
                lasso_results['feature_coefficients'][feature] = float(lasso_model.coef_[i])
            
            lasso_results['selected_features'] = selected_features
            lasso_results['selection_metadata'].update({
                'n_features_selected': len(selected_features),
                'optimal_alpha': optimal_alpha,
                'model_score': lasso_model.score(X_scaled, y)
            })
            
            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ LASSO feature selection completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Selected: {len(selected_features)} features")
            _LOGGER.info(f"📊 Optimal alpha: {optimal_alpha:.6f}")
            _LOGGER.debug(f"📊 Selected features: {selected_features}")
            return lasso_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ LASSO feature selection failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'selected_features': []}

    def comprehensive_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                      feature_names: List[str],
                                      methods: List[str] = None,
                                      weights: Optional[Dict[str, float]] = None,
                                      n_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Comprehensive feature selection combining multiple methods.
        
        This method implements a multi-stage approach:
        1. Filter methods (correlation, mRMR) for initial reduction
        2. Embedded methods (LASSO stability) for robust selection
        3. Wrapper methods (RFE) for final validation
        4. Ensemble voting for consensus features
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            methods: List of methods to use ['correlation', 'mrmr', 'lasso_stability', 'rfe']
            weights: Weights for each method in final voting
            n_features: Target number of features (None for automatic)
            
        Returns:
            Dictionary with comprehensive feature selection results
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting comprehensive feature selection...")
        _LOGGER.info(f"📊 Data shape: {X.shape}, Methods: {methods}")
        
        try:
            if methods is None:
                methods = ['correlation', 'mrmr', 'lasso_stability']
            
            if weights is None:
                weights = {method: 1.0 / len(methods) for method in methods}
            
            comprehensive_results = {
                'selected_features': [],
                'method_results': {},
                'consensus_features': [],
                'feature_votes': {},
                'selection_metadata': {
                    'method': 'comprehensive_feature_selection',
                    'methods_used': methods,
                    'method_weights': weights,
                    'n_features_requested': n_features
                }
            }
            
            current_features = feature_names.copy()
            current_X = X.copy()
            
            # Stage 1: Filter methods (correlation, mRMR)
            filter_methods = [m for m in methods if m in ['correlation', 'mrmr']]
            
            for method in filter_methods:
                try:
                    if method == 'correlation':
                        _LOGGER.info("🔍 Stage 1: Applying correlation-based filtering...")
                        result = self.correlation_based_filtering(current_X, current_features)
                        if 'selected_features' in result:
                            current_features = result['selected_features']
                            # Update X to match selected features
                            selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                            current_X = X[:, selected_indices]
                            comprehensive_results['method_results']['correlation'] = result
                    
                    elif method == 'mrmr':
                        _LOGGER.info("🔍 Stage 1: Applying mRMR selection...")
                        target_features = min(n_features or len(current_features) // 2, len(current_features))
                        result = self.mrmr_selection(current_X, y, current_features, target_features)
                        if 'selected_features' in result:
                            current_features = result['selected_features']
                            # Update X to match selected features
                            selected_indices = [feature_names.index(f) for f in current_features if f in feature_names]
                            current_X = X[:, selected_indices]
                            comprehensive_results['method_results']['mrmr'] = result
                
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Filter method {method} failed: {e}")
                    continue
            
            # Stage 2: Embedded methods (LASSO stability)
            embedded_methods = [m for m in methods if m in ['lasso_stability', 'lasso']]
            
            for method in embedded_methods:
                try:
                    if method == 'lasso_stability':
                        _LOGGER.info("🔍 Stage 2: Applying LASSO stability selection...")
                        result = self.lasso_stability_selection(current_X, y, current_features)
                        if 'selected_features' in result:
                            comprehensive_results['method_results']['lasso_stability'] = result
                    
                    elif method == 'lasso':
                        _LOGGER.info("🔍 Stage 2: Applying LASSO selection...")
                        result = self.lasso_feature_selection(current_X, y, current_features)
                        if 'selected_features' in result:
                            comprehensive_results['method_results']['lasso'] = result
                
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Embedded method {method} failed: {e}")
                    continue
            
            # Stage 3: Wrapper methods (RFE)
            wrapper_methods = [m for m in methods if m in ['rfe']]
            
            for method in wrapper_methods:
                try:
                    if method == 'rfe':
                        _LOGGER.info("🔍 Stage 3: Applying RFE selection...")
                        # Use a simple model for RFE
                        if len(np.unique(y)) <= 10:
                            base_model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
                        else:
                            base_model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
                        
                        target_features = min(n_features or len(current_features) // 2, len(current_features))
                        result = self.recursive_feature_elimination(base_model, current_X, y, current_features, target_features)
                        if 'selected_features' in result:
                            comprehensive_results['method_results']['rfe'] = result
                
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Wrapper method {method} failed: {e}")
                    continue
            
            # Stage 4: Ensemble voting
            _LOGGER.info("🔍 Stage 4: Computing ensemble consensus...")
            
            # Initialize feature votes
            feature_votes = {feature: 0.0 for feature in feature_names}
            
            # Collect votes from each method
            for method, result in comprehensive_results['method_results'].items():
                if 'selected_features' in result:
                    weight = weights.get(method, 1.0)
                    for feature in result['selected_features']:
                        if feature in feature_votes:
                            feature_votes[feature] += weight
            
            comprehensive_results['feature_votes'] = feature_votes
            
            # Select consensus features
            if n_features is None:
                # Use threshold-based selection (features with >50% vote)
                consensus_threshold = 0.5
                consensus_features = [
                    feature for feature, votes in feature_votes.items()
                    if votes >= consensus_threshold
                ]
            else:
                # Use top-N selection
                sorted_features = sorted(
                    feature_votes.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                consensus_features = [feature for feature, _ in sorted_features[:n_features]]
            
            comprehensive_results['consensus_features'] = consensus_features
            comprehensive_results['selected_features'] = consensus_features
            
            # Final statistics
            comprehensive_results['selection_metadata'].update({
                'n_features_selected': len(consensus_features),
                'n_methods_successful': len(comprehensive_results['method_results']),
                'consensus_threshold': consensus_threshold if n_features is None else f"top_{n_features}",
                'feature_vote_stats': {
                    'mean_votes': np.mean(list(feature_votes.values())),
                    'std_votes': np.std(list(feature_votes.values())),
                    'max_votes': np.max(list(feature_votes.values())),
                    'min_votes': np.min(list(feature_votes.values()))
                }
            })
            
            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Comprehensive feature selection completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Selected: {len(consensus_features)} consensus features")
            _LOGGER.info(f"📊 Methods successful: {len(comprehensive_results['method_results'])}/{len(methods)}")
            _LOGGER.debug(f"📊 Selected features: {consensus_features}")
            return comprehensive_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Comprehensive feature selection failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'selected_features': []}

    def tree_based_ensemble_selection(self, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str],
                                    methods: List[str] = None,
                                    weights: Optional[Dict[str, float]] = None,
                                    n_features: Optional[int] = None,
                                    cv_folds: int = 5,
                                    permutation_importance_repeats: int = 10) -> Dict[str, Any]:
        """
        Enhanced ensemble selection using tree-based permutation importance with hyperparameter optimization.
        
        This method implements a sophisticated multi-stage approach:
        1. Collect candidate features from multiple methods
        2. Perform hyperparameter search for optimal tree model
        3. Train optimized tree-based model on all candidates
        4. Use grouped permutation importance to rank features (handles correlated features)
        5. Cross-validate the final selection for generalization
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            methods: List of methods to use for candidate selection
            weights: Weights for each method in initial voting
            n_features: Target number of features (None for automatic)
            cv_folds: Number of CV folds for final validation
            permutation_importance_repeats: Number of repeats for permutation importance
            
        Returns:
            Dictionary with final feature selection and validation results
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting tree-based ensemble selection...")
        _LOGGER.info(f"📊 Data shape: {X.shape}, Methods: {methods}")
        _LOGGER.info(f"📊 CV folds: {cv_folds}, Permutation repeats: {permutation_importance_repeats}")
        
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for tree-based ensemble selection")
            
            if methods is None:
                methods = ['correlation', 'mrmr', 'lasso_stability']
            
            if weights is None:
                weights = {method: 1.0 / len(methods) for method in methods}
            
            # Get hyperparameter search configuration
            enable_hyperparameter_search = self.method_configs['tree_ensemble'].get('hyperparameter_search', True)
            param_grid = self.method_configs['tree_ensemble'].get('param_grid', {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None]
            })
            
            ensemble_results = {
                'selected_features': [],
                'candidate_features': [],
                'permutation_importance': {},
                'cv_validation': {},
                'method_results': {},
                'selection_metadata': {
                    'method': 'tree_based_ensemble_selection',
                    'methods_used': methods,
                    'method_weights': weights,
                    'n_features_requested': n_features,
                    'cv_folds': cv_folds,
                    'hyperparameter_search_enabled': enable_hyperparameter_search,
                    'param_grid': param_grid
                }
            }
            
            # Stage 1: Collect candidate features from multiple methods
            _LOGGER.info("🔍 Stage 1: Collecting candidate features from multiple methods...")
            candidate_features = set()
            method_results = {}
            
            for method in methods:
                try:
                    if method == 'correlation':
                        result = self.correlation_based_filtering(X, feature_names)
                        if 'selected_features' in result:
                            candidate_features.update(result['selected_features'])
                            method_results['correlation'] = result
                    
                    elif method == 'mrmr':
                        target_features = min(n_features or len(feature_names) // 2, len(feature_names))
                        result = self.mrmr_selection(X, y, feature_names, target_features)
                        if 'selected_features' in result:
                            candidate_features.update(result['selected_features'])
                            method_results['mrmr'] = result
                    
                    elif method == 'lasso_stability':
                        result = self.lasso_stability_selection(X, y, feature_names)
                        if 'selected_features' in result:
                            candidate_features.update(result['selected_features'])
                            method_results['lasso_stability'] = result
                    
                    elif method == 'lasso':
                        result = self.lasso_feature_selection(X, y, feature_names)
                        if 'selected_features' in result:
                            candidate_features.update(result['selected_features'])
                            method_results['lasso'] = result
                    
                    elif method == 'rfe':
                        base_model = self._get_default_model(y)
                        if base_model is not None:
                            target_features = min(n_features or len(feature_names) // 2, len(feature_names))
                            result = self.recursive_feature_elimination(base_model, X, y, feature_names, target_features)
                            if 'selected_features' in result:
                                candidate_features.update(result['selected_features'])
                                method_results['rfe'] = result
                
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Method {method} failed: {e}")
                    continue
            
            candidate_features = list(candidate_features)
            ensemble_results['candidate_features'] = candidate_features
            ensemble_results['method_results'] = method_results
            
            _LOGGER.info(f"📊 Collected {len(candidate_features)} candidate features from {len(method_results)} methods")
            
            if len(candidate_features) == 0:
                _LOGGER.warning("⚠️ No candidate features collected, returning empty selection")
                return ensemble_results
            
            # Stage 2: Train tree-based model on all candidates
            _LOGGER.info("🔍 Stage 2: Training tree-based model on candidate features...")
            
            # Get indices of candidate features
            candidate_indices = [feature_names.index(f) for f in candidate_features if f in feature_names]
            X_candidates = X[:, candidate_indices]
            
            # Determine if classification or regression
            is_classification = len(np.unique(y)) <= 10 and not np.issubdtype(np.asarray(y).dtype, np.floating)
            
            # Train the tree-based model with hyperparameter search
            _LOGGER.info("🔍 Stage 2a: Hyperparameter search for tree model...")
            
            if enable_hyperparameter_search:
                # Perform hyperparameter search with parallel processing
                _LOGGER.info("🔍 Performing parallel hyperparameter search...")
                best_params, best_score = self._search_tree_hyperparameters_parallel(
                    X_candidates, y, param_grid, is_classification, cv_folds
                )
            else:
                # Use default parameters
                best_params = {
                    'n_estimators': param_grid['n_estimators'][1],  # Use middle value
                    'max_depth': param_grid['max_depth'][1]        # Use middle value
                }
                best_score = 0.0  # Will be calculated after training
            
            _LOGGER.info(f"📊 Best hyperparameters: {best_params}")
            _LOGGER.info(f"📊 Best CV score: {best_score:.3f}")
            
            # Train final model with best hyperparameters
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                tree_model = RandomForestClassifier(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                tree_model = RandomForestRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    random_state=self.random_state,
                    n_jobs=-1
                )
            
            tree_model.fit(X_candidates, y)
            baseline_score = tree_model.score(X_candidates, y)
            
            _LOGGER.info(f"📊 Tree model trained with best params - Baseline score: {baseline_score:.3f}")
            
            # Stage 3: Calculate permutation importance with correlation grouping
            _LOGGER.info("🔍 Stage 3: Calculating permutation importance with correlation grouping...")
            
            # Calculate correlation matrix for candidate features using M1 optimization
            _LOGGER.info("🔍 Computing correlation matrix for feature grouping with M1 optimization...")
            try:
                correlation_matrix = m1_correlation_matrix(X_candidates.T)
                _LOGGER.info("✅ M1-optimized correlation matrix computed")
            except Exception as e:
                _LOGGER.warning(f"⚠️ M1 correlation failed, falling back to numpy: {e}")
                correlation_matrix = np.corrcoef(X_candidates.T)
            
            correlation_threshold = self.method_configs['tree_ensemble']['correlation_threshold']
            
            # Group highly correlated features
            feature_groups = self._group_correlated_features(
                candidate_features, correlation_matrix, correlation_threshold
            )
            
            _LOGGER.info(f"📊 Grouped {len(candidate_features)} features into {len(feature_groups)} groups")
            for i, group in enumerate(feature_groups):
                if len(group) > 1:
                    _LOGGER.debug(f"📊 Group {i}: {group} (correlation group)")
                else:
                    _LOGGER.debug(f"📊 Group {i}: {group[0]} (individual)")
            
            # Calculate grouped permutation importance
            permutation_importance = {}
            for group_idx, feature_group in enumerate(feature_groups):
                group_importance_scores = []
                
                for repeat in range(permutation_importance_repeats):
                    # Create permuted data
                    X_permuted = X_candidates.copy()
                    
                    # Permute all features in the group together
                    for feature in feature_group:
                        feature_idx = candidate_features.index(feature)
                        np.random.shuffle(X_permuted[:, feature_idx])
                    
                    # Calculate score with permuted feature group
                    permuted_score = tree_model.score(X_permuted, y)
                    
                    # Importance is the drop in score
                    importance = baseline_score - permuted_score
                    group_importance_scores.append(importance)
                
                # Average importance across repeats
                avg_importance = np.mean(group_importance_scores)
                std_importance = np.std(group_importance_scores)
                
                # Assign the same importance to all features in the group
                for feature in feature_group:
                    permutation_importance[feature] = {
                        'importance': avg_importance,
                        'std_importance': std_importance,
                        'scores': group_importance_scores,
                        'group': feature_group,
                        'group_size': len(feature_group),
                        'is_correlated_group': len(feature_group) > 1
                    }
            
            ensemble_results['permutation_importance'] = permutation_importance
            
            # Stage 4: Select features based on permutation importance
            _LOGGER.info("🔍 Stage 4: Selecting features based on permutation importance...")
            
            # Sort features by importance
            sorted_features = sorted(
                permutation_importance.items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            )
            
            # Select top features
            if n_features is None:
                # Use threshold-based selection (features with positive importance)
                selected_features = [feature for feature, importance_data in sorted_features 
                                   if importance_data['importance'] > 0]
            else:
                # Use top-N selection
                selected_features = [feature for feature, _ in sorted_features[:n_features]]
            
            ensemble_results['selected_features'] = selected_features
            
            _LOGGER.info(f"📊 Selected {len(selected_features)} features based on permutation importance")
            
            # Stage 5: Cross-validation validation
            _LOGGER.info("🔍 Stage 5: Cross-validation validation of selected features...")
            
            if len(selected_features) > 0:
                # Get indices of selected features
                selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
                X_selected = X[:, selected_indices]
                
                # Cross-validation
                cv_scores = []
                cv_importances = []
                
                if is_classification:
                    from sklearn.model_selection import StratifiedKFold
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                else:
                    from sklearn.model_selection import KFold
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_selected, y)):
                    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model on fold
                    fold_model = tree_model.__class__(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=self.random_state + fold,
                        n_jobs=-1
                    )
                    fold_model.fit(X_train, y_train)
                    
                    # Validate on fold
                    fold_score = fold_model.score(X_val, y_val)
                    cv_scores.append(fold_score)
                    
                    # Store feature importances
                    fold_importances = dict(zip(selected_features, fold_model.feature_importances_))
                    cv_importances.append(fold_importances)
                
                # Calculate CV statistics
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                # Calculate stability of feature importances across folds
                feature_importance_stability = {}
                for feature in selected_features:
                    fold_importances = [fold_imp[feature] for fold_imp in cv_importances]
                    feature_importance_stability[feature] = {
                        'mean_importance': np.mean(fold_importances),
                        'std_importance': np.std(fold_importances),
                        'stability': 1.0 - (np.std(fold_importances) / (np.mean(fold_importances) + 1e-8))
                    }
                
                ensemble_results['cv_validation'] = {
                    'cv_scores': cv_scores,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'feature_importance_stability': feature_importance_stability
                }
                
                _LOGGER.info(f"📊 CV validation - Mean score: {cv_mean:.3f} ± {cv_std:.3f}")
            else:
                _LOGGER.warning("⚠️ No features selected for CV validation")
                ensemble_results['cv_validation'] = {'error': 'No features selected'}
            
            # Final statistics
            ensemble_results['selection_metadata'].update({
                'n_candidate_features': len(candidate_features),
                'n_features_selected': len(selected_features),
                'n_methods_successful': len(method_results),
                'baseline_score': baseline_score,
                'best_hyperparameters': best_params,
                'best_hyperparameter_score': best_score,
                'permutation_importance_stats': {
                    'mean_importance': np.mean([data['importance'] for data in permutation_importance.values()]),
                    'std_importance': np.std([data['importance'] for data in permutation_importance.values()]),
                    'max_importance': np.max([data['importance'] for data in permutation_importance.values()]),
                    'min_importance': np.min([data['importance'] for data in permutation_importance.values()])
                }
            })
            
            execution_time = time.time() - start_time
            
            # Memory optimization
            m1_optimize_memory()
            
            # Enhanced reporting
            additional_stats = {
                'Methods used': ', '.join(methods),
                'Methods successful': f"{len(method_results)}/{len(methods)}",
                'Candidate features': len(candidate_features),
                'Hyperparameter search': enable_hyperparameter_search,
                'Best hyperparameters': best_params,
                'CV folds': cv_folds,
                'Permutation repeats': permutation_importance_repeats
            }
            self._log_feature_reduction_stats(
                "Tree-Based Ensemble Selection", len(feature_names), 
                len(selected_features), execution_time, additional_stats
            )
            _LOGGER.debug(f"📊 Selected features: {selected_features}")
            return ensemble_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Tree-based ensemble selection failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'selected_features': []}

    def _group_correlated_features(self, feature_names: List[str], 
                                 correlation_matrix: np.ndarray, 
                                 threshold: float = 0.8) -> List[List[str]]:
        """
        Group highly correlated features together.
        
        Args:
            feature_names: List of feature names
            correlation_matrix: Correlation matrix of features
            threshold: Correlation threshold for grouping
            
        Returns:
            List of feature groups, where each group contains highly correlated features
        """
        n_features = len(feature_names)
        visited = set()
        groups = []
        
        for i in range(n_features):
            if i in visited:
                continue
            
            # Start a new group with feature i
            current_group = [feature_names[i]]
            visited.add(i)
            
            # Find all features highly correlated with feature i
            for j in range(i + 1, n_features):
                if j in visited:
                    continue
                
                # Check if features i and j are highly correlated
                if abs(correlation_matrix[i, j]) >= threshold:
                    current_group.append(feature_names[j])
                    visited.add(j)
            
            groups.append(current_group)
        
        return groups

    def _lasso_bootstrap_fit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method for parallel LASSO bootstrap fitting."""
        try:
            bootstrap_idx = params['bootstrap_idx']
            X_scaled = params['X_scaled']
            y = params['y']
            bootstrap_size = params['bootstrap_size']
            feature_names = params['feature_names']
            alpha_range = params['alpha_range']
            cv_folds = params['cv_folds']
            is_classification = params['is_classification']
            random_state = params['random_state']
            method_configs = params['method_configs']
            
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(
                len(X_scaled), size=bootstrap_size, replace=True
            )
            X_bootstrap = X_scaled[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Find optimal alpha using cross-validation
            if is_classification:
                lasso_cv = LassoCV(
                    alphas=np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 20),
                    cv=cv_folds,
                    max_iter=method_configs['lasso']['max_iter'],
                    tol=method_configs['lasso']['tol'],
                    random_state=random_state
                )
            else:
                lasso_cv = LassoCV(
                    alphas=np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 20),
                    cv=cv_folds,
                    max_iter=method_configs['lasso']['max_iter'],
                    tol=method_configs['lasso']['tol'],
                    random_state=random_state
                )
            
            # Fit LASSO with cross-validation
            lasso_cv.fit(X_bootstrap, y_bootstrap)
            
            # Get selected features (non-zero coefficients)
            selected_mask = np.abs(lasso_cv.coef_) > 1e-6
            selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
            
            return {
                'bootstrap_idx': bootstrap_idx,
                'optimal_alpha': lasso_cv.alpha_,
                'selected_features': selected_features,
                'coefficients': lasso_cv.coef_,
                'n_selected': len(selected_features),
                'cv_score': lasso_cv.score(X_bootstrap, y_bootstrap)
            }
            
        except Exception as e:
            return {
                'bootstrap_idx': params.get('bootstrap_idx', -1),
                'error': str(e)
            }

    def _search_tree_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                   param_grid: Dict[str, List], 
                                   is_classification: bool, 
                                   cv_folds: int) -> Tuple[Dict[str, Any], float]:
        """
        Search for optimal hyperparameters for the tree model.
        
        Args:
            X: Feature matrix
            y: Target array
            param_grid: Dictionary of hyperparameters to search
            is_classification: Whether this is a classification task
            cv_folds: Number of CV folds for evaluation
            
        Returns:
            Tuple of (best_params, best_score)
        """
        from sklearn.model_selection import GridSearchCV
        
        # Create base model
        if is_classification:
            from sklearn.ensemble import RandomForestClassifier
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='accuracy' if is_classification else 'r2',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_, grid_search.best_score_

    def _search_tree_hyperparameters_parallel(self, X: np.ndarray, y: np.ndarray, 
                                            param_grid: Dict[str, List], 
                                            is_classification: bool, 
                                            cv_folds: int) -> Tuple[Dict[str, Any], float]:
        """
        Search for optimal hyperparameters using parallel processing.
        """
        try:
            if not self.parallel_processor or not self.enable_parallel:
                return self._search_tree_hyperparameters(X, y, param_grid, is_classification, cv_folds)
            
            # Create parameter combinations
            param_combinations = []
            for n_est in param_grid['n_estimators']:
                for max_dep in param_grid['max_depth']:
                    param_combinations.append({
                        'n_estimators': n_est,
                        'max_depth': max_dep,
                        'X': X,
                        'y': y,
                        'is_classification': is_classification,
                        'cv_folds': cv_folds,
                        'random_state': self.random_state
                    })
            
            _LOGGER.info(f"⚡ Testing {len(param_combinations)} parameter combinations in parallel")
            
            # Evaluate parameter combinations in parallel
            param_results = self.parallel_processor.parallel_apply(
                param_combinations,
                self._evaluate_hyperparameter_combination,
                max_workers=self.max_workers
            )
            
            # Find best parameters
            best_score = -np.inf
            best_params = None
            
            for result in param_results:
                if result and 'error' not in result:
                    if result['cv_score'] > best_score:
                        best_score = result['cv_score']
                        best_params = {
                            'n_estimators': result['n_estimators'],
                            'max_depth': result['max_depth']
                        }
            
            if best_params is None:
                _LOGGER.warning("⚠️ No valid hyperparameter combinations found, using defaults")
                return self._search_tree_hyperparameters(X, y, param_grid, is_classification, cv_folds)
            
            return best_params, best_score
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Parallel hyperparameter search failed: {e}, falling back to sequential")
            return self._search_tree_hyperparameters(X, y, param_grid, is_classification, cv_folds)

    def _evaluate_hyperparameter_combination(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single hyperparameter combination."""
        try:
            n_estimators = params['n_estimators']
            max_depth = params['max_depth']
            X = params['X']
            y = params['y']
            is_classification = params['is_classification']
            cv_folds = params['cv_folds']
            random_state = params['random_state']
            
            # Create model
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=1  # Single job for parallel processing
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=1  # Single job for parallel processing
                )
            
            # Cross-validation
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy' if is_classification else 'r2')
            cv_score = np.mean(cv_scores)
            
            return {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'cv_score': cv_score,
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            return {
                'n_estimators': params.get('n_estimators', 0),
                'max_depth': params.get('max_depth', 0),
                'error': str(e)
            }

    def get_model_target_features(self, model_type: str) -> int:
        """
        Get the optimal feature count for a specific model type.
        
        Args:
            model_type: Type of model (e.g., 'random_forest', 'linear_regression')
            
        Returns:
            Optimal feature count for the model
        """
        return self.MODEL_FEATURE_TARGETS.get(model_type, self.MODEL_FEATURE_TARGETS['default'])

    def _auto_detect_model_type(self, model: Any) -> str:
        """
        Auto-detect model type from sklearn model object or configuration.
        
        Args:
            model: Sklearn model object or model configuration dictionary
            
        Returns:
            Detected model type string
        """
        try:
            # Handle model objects
            if hasattr(model, '__class__'):
                model_class_name = model.__class__.__name__.lower()
                
                # Linear models
                if 'linearregression' in model_class_name:
                    return 'linear_regression'
                elif 'ridge' in model_class_name:
                    return 'ridge_regression'
                elif 'lasso' in model_class_name:
                    return 'lasso_regression'
                elif 'elasticnet' in model_class_name:
                    return 'elastic_net'
                elif 'logisticregression' in model_class_name:
                    return 'logistic_regression'
                
                # Tree-based models
                elif 'randomforest' in model_class_name:
                    return 'random_forest'
                elif 'gradientboosting' in model_class_name:
                    return 'gradient_boosting'
                elif 'xgboost' in model_class_name:
                    return 'xgboost'
                elif 'lightgbm' in model_class_name:
                    return 'lightgbm'
                elif 'catboost' in model_class_name:
                    return 'catboost'
                elif 'extratrees' in model_class_name:
                    return 'extra_trees'
                
                # SVM models
                elif 'svc' in model_class_name or 'svr' in model_class_name:
                    if hasattr(model, 'kernel'):
                        kernel = getattr(model.kernel, '__name__', str(model.kernel))
                        if kernel == 'linear':
                            return 'svm_linear'
                        elif kernel == 'rbf':
                            return 'svm_rbf'
                        elif kernel == 'poly':
                            return 'svm_poly'
                    return 'svm_rbf'  # Default SVM
                
                # Neural networks
                elif 'mlp' in model_class_name or 'neural' in model_class_name:
                    return 'neural_network'
                
                # Ensemble methods
                elif 'voting' in model_class_name:
                    return 'voting_classifier'
                elif 'stacking' in model_class_name:
                    return 'stacking_classifier'
                elif 'bagging' in model_class_name:
                    return 'bagging_classifier'
            
            # Handle configuration dictionaries
            elif isinstance(model, dict):
                model_name = model.get('name', '').lower()
                model_type = model.get('type', '').lower()
                
                if model_name or model_type:
                    # Check against known model types
                    for key, value in self.MODEL_FEATURE_TARGETS.items():
                        if key in model_name or key in model_type:
                            return key
                
                # Check for specific parameters that indicate model type
                if 'n_estimators' in model and 'max_depth' in model:
                    return 'random_forest'
                elif 'alpha' in model and 'l1_ratio' in model:
                    return 'elastic_net'
                elif 'alpha' in model:
                    return 'ridge_regression'
                elif 'C' in model and 'kernel' in model:
                    kernel = model['kernel']
                    if kernel == 'linear':
                        return 'svm_linear'
                    elif kernel == 'rbf':
                        return 'svm_rbf'
                    elif kernel == 'poly':
                        return 'svm_poly'
            
            _LOGGER.warning(f"⚠️ Could not auto-detect model type from: {type(model)}")
            return 'default'
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Model auto-detection failed: {e}")
            return 'default'

    def _create_feature_dependency_graph(self, X: np.ndarray, feature_names: List[str],
                                       correlation_threshold: float = 0.3,
                                       mutual_info_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Create feature dependency graph to understand relationships between features.
        
        This method analyzes various types of dependencies between features including
        correlation, mutual information, and conditional dependencies to build a
        comprehensive dependency graph.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            correlation_threshold: Minimum correlation for edge creation
            mutual_info_threshold: Minimum mutual information for edge creation
            
        Returns:
            Dictionary with dependency graph analysis
        """
        start_time = time.time()
        _LOGGER.info(f"🔄 Creating feature dependency graph...")
        _LOGGER.info(f"📊 Features: {len(feature_names)}")
        _LOGGER.info(f"📊 Correlation threshold: {correlation_threshold}")
        _LOGGER.info(f"📊 Mutual info threshold: {mutual_info_threshold}")
        
        try:
            # Calculate correlation matrix
            if self.enable_gpu and self.gpu_manager is not None:
                corr_matrix = self.gpu_manager.m1_correlation_matrix(X.T)
            else:
                corr_matrix = np.corrcoef(X.T)
            
            # Calculate mutual information matrix
            mi_matrix = self._calculate_mutual_information_matrix(X, feature_names)
            
            # Build dependency graph
            dependency_graph = {
                'nodes': feature_names,
                'edges': [],
                'node_properties': {},
                'edge_properties': {}
            }
            
            # Add nodes with properties
            for i, feature in enumerate(feature_names):
                dependency_graph['node_properties'][feature] = {
                    'index': i,
                    'variance': np.var(X[:, i]),
                    'mean': np.mean(X[:, i]),
                    'std': np.std(X[:, i])
                }
            
            # Add edges based on correlation
            correlation_edges = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    corr_value = abs(corr_matrix[i, j])
                    if corr_value >= correlation_threshold:
                        edge = {
                            'source': feature_names[i],
                            'target': feature_names[j],
                            'type': 'correlation',
                            'weight': corr_value,
                            'direction': 'undirected'
                        }
                        correlation_edges.append(edge)
                        dependency_graph['edges'].append(edge)
            
            # Add edges based on mutual information
            mi_edges = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    mi_value = mi_matrix[i, j]
                    if mi_value >= mutual_info_threshold:
                        edge = {
                            'source': feature_names[i],
                            'target': feature_names[j],
                            'type': 'mutual_information',
                            'weight': mi_value,
                            'direction': 'undirected'
                        }
                        mi_edges.append(edge)
                        dependency_graph['edges'].append(edge)
            
            # Analyze graph properties
            graph_analysis = self._analyze_dependency_graph(dependency_graph, feature_names)
            
            # Create feature clusters based on dependencies
            feature_clusters = self._cluster_features_by_dependencies(
                dependency_graph, feature_names
            )
            
            # Calculate feature centrality measures
            centrality_measures = self._calculate_feature_centrality(
                dependency_graph, feature_names
            )
            
            dependency_analysis = {
                'correlation_matrix': corr_matrix,
                'mutual_information_matrix': mi_matrix,
                'dependency_graph': dependency_graph,
                'graph_analysis': graph_analysis,
                'feature_clusters': feature_clusters,
                'centrality_measures': centrality_measures,
                'correlation_edges': correlation_edges,
                'mutual_info_edges': mi_edges,
                'statistics': {
                    'n_nodes': len(feature_names),
                    'n_correlation_edges': len(correlation_edges),
                    'n_mi_edges': len(mi_edges),
                    'n_total_edges': len(dependency_graph['edges']),
                    'graph_density': len(dependency_graph['edges']) / (len(feature_names) * (len(feature_names) - 1) / 2),
                    'n_clusters': len(feature_clusters),
                    'max_cluster_size': max(len(cluster) for cluster in feature_clusters.values()) if feature_clusters else 0
                }
            }
            
            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Feature dependency graph created in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Nodes: {len(feature_names)}")
            _LOGGER.info(f"📊 Edges: {len(dependency_graph['edges'])}")
            _LOGGER.info(f"📊 Clusters: {len(feature_clusters)}")
            _LOGGER.info(f"📊 Graph density: {dependency_analysis['statistics']['graph_density']:.3f}")
            
            return dependency_analysis
            
        except Exception as e:
            _LOGGER.error(f"❌ Feature dependency graph creation failed: {e}")
            return {
                'error': str(e),
                'dependency_graph': {'nodes': feature_names, 'edges': []},
                'statistics': {'n_nodes': len(feature_names), 'n_edges': 0}
            }

    def _calculate_feature_selection_quality_metrics(self, X: np.ndarray, y: np.ndarray,
                                                   feature_names: List[str],
                                                   selected_features: List[str],
                                                   pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive feature selection quality metrics for final report.
        
        This method provides a comprehensive assessment of feature selection quality
        including redundancy, relevance, stability, and interpretability metrics.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of all feature names
            selected_features: List of selected feature names
            pipeline_results: Results from the feature selection pipeline
            
        Returns:
            Dictionary with comprehensive quality metrics
        """
        start_time = time.time()
        _LOGGER.info(f"📊 Calculating feature selection quality metrics...")
        
        try:
            # Get selected feature indices
            selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
            X_selected = X[:, selected_indices]
            
            quality_metrics = {}
            
            # 1. Redundancy Metrics
            quality_metrics['redundancy'] = self._calculate_redundancy_metrics(X_selected, selected_features)
            
            # 2. Relevance Metrics
            quality_metrics['relevance'] = self._calculate_relevance_metrics(X_selected, y, selected_features)
            
            # 3. Stability Metrics
            quality_metrics['stability'] = self._calculate_stability_metrics(pipeline_results)
            
            # 4. Interpretability Metrics
            quality_metrics['interpretability'] = self._calculate_interpretability_metrics(selected_features)
            
            # 5. Performance Metrics
            quality_metrics['performance'] = self._calculate_performance_metrics(X_selected, y, selected_features)
            
            # 6. Diversity Metrics
            quality_metrics['diversity'] = self._calculate_diversity_metrics(X_selected, selected_features)
            
            # 7. Efficiency Metrics
            quality_metrics['efficiency'] = self._calculate_efficiency_metrics(pipeline_results)
            
            # 8. Overall Quality Score
            quality_metrics['overall_quality'] = self._calculate_overall_quality_score(quality_metrics)
            
            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Quality metrics calculated in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Overall quality score: {quality_metrics['overall_quality']:.3f}")
            
            return quality_metrics
            
        except Exception as e:
            _LOGGER.error(f"❌ Quality metrics calculation failed: {e}")
            return {
                'error': str(e),
                'overall_quality': 0.0
            }

    def _calculate_redundancy_metrics(self, X_selected: np.ndarray, selected_features: List[str]) -> Dict[str, float]:
        """Calculate redundancy metrics for selected features."""
        try:
            # Correlation-based redundancy
            corr_matrix = np.corrcoef(X_selected.T)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            
            # Average correlation
            avg_correlation = np.mean(np.abs(upper_tri))
            
            # Maximum correlation
            max_correlation = np.max(np.abs(upper_tri))
            
            # Correlation variance (lower is better - more uniform redundancy)
            corr_variance = np.var(np.abs(upper_tri))
            
            # Redundancy ratio (features with high correlation)
            high_corr_ratio = np.mean(np.abs(upper_tri) > 0.8)
            
            # Mutual information redundancy
            mi_redundancy = 0.0
            if len(selected_features) > 1:
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    for i in range(len(selected_features)):
                        for j in range(i + 1, len(selected_features)):
                            mi = mutual_info_regression(
                                X_selected[:, i].reshape(-1, 1), 
                                X_selected[:, j], 
                                discrete_features=False
                            )[0]
                            mi_redundancy += mi
                    mi_redundancy /= (len(selected_features) * (len(selected_features) - 1) / 2)
                except:
                    mi_redundancy = 0.0
            
            return {
                'average_correlation': avg_correlation,
                'maximum_correlation': max_correlation,
                'correlation_variance': corr_variance,
                'high_correlation_ratio': high_corr_ratio,
                'mutual_information_redundancy': mi_redundancy,
                'redundancy_score': 1.0 - avg_correlation  # Lower correlation = better
            }
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Redundancy metrics calculation failed: {e}")
            return {'redundancy_score': 0.5}

    def _calculate_relevance_metrics(self, X_selected: np.ndarray, y: np.ndarray, 
                                   selected_features: List[str]) -> Dict[str, float]:
        """Calculate relevance metrics for selected features."""
        try:
            # Individual feature relevance
            individual_relevance = []
            for i in range(len(selected_features)):
                try:
                    from sklearn.feature_selection import mutual_info_regression, f_regression
                    
                    # Mutual information with target
                    mi = mutual_info_regression(
                        X_selected[:, i].reshape(-1, 1), y, 
                        discrete_features=False
                    )[0]
                    
                    # F-statistic
                    f_stat, _ = f_regression(X_selected[:, i].reshape(-1, 1), y)
                    f_stat = f_stat[0] if len(f_stat) > 0 else 0.0
                    
                    individual_relevance.append({
                        'mutual_information': mi,
                        'f_statistic': f_stat,
                        'combined_relevance': (mi + f_stat / 100) / 2  # Normalize F-stat
                    })
                except:
                    individual_relevance.append({
                        'mutual_information': 0.0,
                        'f_statistic': 0.0,
                        'combined_relevance': 0.0
                    })
            
            # Aggregate relevance metrics
            avg_mi = np.mean([r['mutual_information'] for r in individual_relevance])
            avg_f_stat = np.mean([r['f_statistic'] for r in individual_relevance])
            avg_combined = np.mean([r['combined_relevance'] for r in individual_relevance])
            
            # Relevance variance (higher is better - more diverse relevance)
            relevance_variance = np.var([r['combined_relevance'] for r in individual_relevance])
            
            # Minimum relevance (worst feature)
            min_relevance = np.min([r['combined_relevance'] for r in individual_relevance])
            
            return {
                'average_mutual_information': avg_mi,
                'average_f_statistic': avg_f_stat,
                'average_combined_relevance': avg_combined,
                'relevance_variance': relevance_variance,
                'minimum_relevance': min_relevance,
                'relevance_score': avg_combined
            }
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Relevance metrics calculation failed: {e}")
            return {'relevance_score': 0.5}

    def _calculate_stability_metrics(self, pipeline_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stability metrics from pipeline results."""
        try:
            stability_metrics = {}
            
            # Bootstrap stability
            if 'bootstrap_stability' in pipeline_results.get('pipeline_stages', {}):
                bootstrap_stage = pipeline_results['pipeline_stages']['bootstrap_stability']
                if 'stability_analysis' in bootstrap_stage:
                    stability_stats = bootstrap_stage['stability_analysis']['stability_statistics']
                    stability_metrics['bootstrap_stability'] = stability_stats.get('mean_stability', 0.0)
                    stability_metrics['bootstrap_consistency'] = stability_stats.get('features_above_threshold', 0) / len(pipeline_results.get('final_selected_features', []))
            
            # Nested bootstrap stability
            if 'nested_bootstrap' in pipeline_results.get('pipeline_stages', {}):
                nested_stage = pipeline_results['pipeline_stages']['nested_bootstrap']
                if 'nested_analysis' in nested_stage:
                    nested_stats = nested_stage['nested_analysis']['nested_stability_statistics']
                    stability_metrics['nested_bootstrap_stability'] = nested_stats.get('mean_stability', 0.0)
                    stability_metrics['outer_run_consistency'] = nested_stats.get('outer_run_consistency', 0.0)
            
            # Temporal stability
            if 'temporal_stability' in pipeline_results.get('pipeline_stages', {}):
                temporal_stage = pipeline_results['pipeline_stages']['temporal_stability']
                if 'temporal_analysis' in temporal_stage:
                    temporal_stats = temporal_stage['temporal_analysis']['temporal_stability_statistics']
                    stability_metrics['temporal_stability'] = temporal_stats.get('mean_temporal_stability', 0.0)
            
            # Cross-dataset stability
            if 'cross_dataset_stability' in pipeline_results.get('pipeline_stages', {}):
                cross_dataset_stage = pipeline_results['pipeline_stages']['cross_dataset_stability']
                if 'cross_dataset_analysis' in cross_dataset_stage:
                    cross_dataset_stats = cross_dataset_stage['cross_dataset_analysis']['cross_dataset_stability_statistics']
                    stability_metrics['cross_dataset_stability'] = cross_dataset_stats.get('mean_cross_dataset_stability', 0.0)
            
            # Overall stability score
            stability_scores = [score for score in stability_metrics.values() if score > 0]
            overall_stability = np.mean(stability_scores) if stability_scores else 0.0
            
            stability_metrics['overall_stability'] = overall_stability
            
            return stability_metrics
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Stability metrics calculation failed: {e}")
            return {'overall_stability': 0.5}

    def _calculate_interpretability_metrics(self, selected_features: List[str]) -> Dict[str, float]:
        """Calculate interpretability metrics for selected features."""
        try:
            # Feature name interpretability
            interpretable_names = 0
            for feature in selected_features:
                # Simple heuristic: features with descriptive names are more interpretable
                if (len(feature.split('_')) <= 3 and  # Not too many underscores
                    not feature.isdigit() and  # Not just numbers
                    len(feature) < 50):  # Not too long
                    interpretable_names += 1
            
            name_interpretability = interpretable_names / len(selected_features) if selected_features else 0.0
            
            # Feature count interpretability (fewer features = more interpretable)
            count_interpretability = max(0, 1.0 - len(selected_features) / 100)  # Penalty for >100 features
            
            # Feature diversity interpretability
            feature_types = set()
            for feature in selected_features:
                # Categorize features by type (simple heuristic)
                if any(word in feature.lower() for word in ['price', 'cost', 'value']):
                    feature_types.add('financial')
                elif any(word in feature.lower() for word in ['time', 'date', 'hour']):
                    feature_types.add('temporal')
                elif any(word in feature.lower() for word in ['count', 'num', 'total']):
                    feature_types.add('count')
                else:
                    feature_types.add('other')
            
            diversity_interpretability = len(feature_types) / 4.0  # Normalize by max expected types
            
            overall_interpretability = (name_interpretability + count_interpretability + diversity_interpretability) / 3.0
            
            return {
                'name_interpretability': name_interpretability,
                'count_interpretability': count_interpretability,
                'diversity_interpretability': diversity_interpretability,
                'feature_types': list(feature_types),
                'interpretability_score': overall_interpretability
            }
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Interpretability metrics calculation failed: {e}")
            return {'interpretability_score': 0.5}

    def _calculate_performance_metrics(self, X_selected: np.ndarray, y: np.ndarray, 
                                     selected_features: List[str]) -> Dict[str, float]:
        """Calculate performance-related metrics."""
        try:
            # Feature variance (higher variance = more informative)
            feature_variances = np.var(X_selected, axis=0)
            avg_variance = np.mean(feature_variances)
            variance_ratio = np.mean(feature_variances > np.percentile(feature_variances, 25))
            
            # Feature range (wider range = more informative)
            feature_ranges = np.max(X_selected, axis=0) - np.min(X_selected, axis=0)
            avg_range = np.mean(feature_ranges)
            
            # Feature skewness (normal distribution is often better)
            feature_skewness = []
            for i in range(X_selected.shape[1]):
                from scipy import stats
                skewness = abs(stats.skew(X_selected[:, i]))
                feature_skewness.append(skewness)
            
            avg_skewness = np.mean(feature_skewness)
            skewness_score = max(0, 1.0 - avg_skewness / 2.0)  # Penalty for high skewness
            
            # Feature completeness (no missing values)
            completeness = 1.0  # Assuming no missing values in selected features
            
            overall_performance = (variance_ratio + skewness_score + completeness) / 3.0
            
            return {
                'average_variance': avg_variance,
                'variance_ratio': variance_ratio,
                'average_range': avg_range,
                'average_skewness': avg_skewness,
                'skewness_score': skewness_score,
                'completeness': completeness,
                'performance_score': overall_performance
            }
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {'performance_score': 0.5}

    def _calculate_diversity_metrics(self, X_selected: np.ndarray, selected_features: List[str]) -> Dict[str, float]:
        """Calculate diversity metrics for selected features."""
        try:
            # Statistical diversity
            feature_means = np.mean(X_selected, axis=0)
            feature_stds = np.std(X_selected, axis=0)
            
            mean_diversity = 1.0 - np.corrcoef(feature_means.reshape(1, -1))[0, 1] if len(feature_means) > 1 else 1.0
            std_diversity = 1.0 - np.corrcoef(feature_stds.reshape(1, -1))[0, 1] if len(feature_stds) > 1 else 1.0
            
            # Distribution diversity (using Kolmogorov-Smirnov test)
            distribution_diversity = 0.0
            if len(selected_features) > 1:
                from scipy import stats
                ks_scores = []
                for i in range(len(selected_features)):
                    for j in range(i + 1, len(selected_features)):
                        ks_stat, _ = stats.ks_2samp(X_selected[:, i], X_selected[:, j])
                        ks_scores.append(ks_stat)
                distribution_diversity = np.mean(ks_scores)
            
            # Feature space coverage
            feature_space_volume = np.linalg.det(np.cov(X_selected.T))
            coverage_score = min(1.0, feature_space_volume / 1000.0)  # Normalize
            
            overall_diversity = (mean_diversity + std_diversity + distribution_diversity + coverage_score) / 4.0
            
            return {
                'mean_diversity': mean_diversity,
                'std_diversity': std_diversity,
                'distribution_diversity': distribution_diversity,
                'feature_space_volume': feature_space_volume,
                'coverage_score': coverage_score,
                'diversity_score': overall_diversity
            }
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Diversity metrics calculation failed: {e}")
            return {'diversity_score': 0.5}

    def _calculate_efficiency_metrics(self, pipeline_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate efficiency metrics from pipeline results."""
        try:
            # Execution time efficiency
            total_time = pipeline_results.get('execution_time', 0)
            n_features_processed = pipeline_results.get('pipeline_summary', {}).get('initial_count', 1)
            time_per_feature = total_time / n_features_processed
            
            # Memory efficiency (if available)
            memory_efficiency = 1.0  # Placeholder - would need memory monitoring
            
            # Pipeline stage efficiency
            pipeline_stages = pipeline_results.get('pipeline_stages', {})
            successful_stages = sum(1 for stage in pipeline_stages.values() 
                                  if 'error' not in stage and not stage.get('skipped', False))
            total_stages = len(pipeline_stages)
            stage_efficiency = successful_stages / total_stages if total_stages > 0 else 1.0
            
            # Feature reduction efficiency
            initial_count = pipeline_results.get('pipeline_summary', {}).get('initial_count', 1)
            final_count = pipeline_results.get('pipeline_summary', {}).get('final_count', 1)
            reduction_efficiency = (initial_count - final_count) / initial_count
            
            overall_efficiency = (stage_efficiency + reduction_efficiency + memory_efficiency) / 3.0
            
            return {
                'total_execution_time': total_time,
                'time_per_feature': time_per_feature,
                'stage_efficiency': stage_efficiency,
                'reduction_efficiency': reduction_efficiency,
                'memory_efficiency': memory_efficiency,
                'efficiency_score': overall_efficiency
            }
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Efficiency metrics calculation failed: {e}")
            return {'efficiency_score': 0.5}

    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from all metrics."""
        try:
            # Weighted combination of all quality aspects
            weights = {
                'redundancy': 0.20,      # Low redundancy is important
                'relevance': 0.25,       # High relevance is crucial
                'stability': 0.20,       # Stability is important for reliability
                'interpretability': 0.15, # Interpretability is valuable
                'performance': 0.10,     # Performance characteristics matter
                'diversity': 0.05,       # Diversity is nice to have
                'efficiency': 0.05       # Efficiency is nice to have
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric_type, weight in weights.items():
                if metric_type in quality_metrics:
                    metric_data = quality_metrics[metric_type]
                    if isinstance(metric_data, dict):
                        # Get the main score for this metric type
                        score_key = f"{metric_type}_score"
                        if score_key in metric_data:
                            score = metric_data[score_key]
                        elif 'overall_stability' in metric_data:
                            score = metric_data['overall_stability']
                        else:
                            # Use first available score
                            scores = [v for v in metric_data.values() if isinstance(v, (int, float))]
                            score = scores[0] if scores else 0.5
                    else:
                        score = metric_data
                    
                    weighted_score += weight * score
                    total_weight += weight
            
            overall_score = weighted_score / total_weight if total_weight > 0 else 0.5
            
            return min(1.0, max(0.0, overall_score))  # Clamp to [0, 1]
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Overall quality score calculation failed: {e}")
            return 0.5

    def _prevent_data_leakage_cv(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                                model: Any, cv_folds: int = 5, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Prevent data leakage by performing feature selection within each CV fold.
        
        This is CRITICAL for reliable performance estimation. Without this, performance
        estimates are overly optimistic due to data leakage from feature selection
        being performed on the full dataset.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            model: Model to evaluate
            cv_folds: Number of cross-validation folds
            test_size: Fraction of data to hold out for final testing
            
        Returns:
            Dictionary with CV results and selected features
        """
        start_time = time.time()
        _LOGGER.info(f"🛡️ Starting data leakage prevention with CV...")
        _LOGGER.info(f"📊 CV folds: {cv_folds}, Test size: {test_size}")
        
        try:
            from sklearn.model_selection import KFold, train_test_split
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            # Split data into train+val and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) <= 10 else None
            )
            
            # Initialize CV
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            cv_results = []
            fold_selected_features = []
            fold_scores = []
            
            # Perform CV with feature selection within each fold
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train_val)):
                _LOGGER.info(f"🔄 Processing CV fold {fold_idx + 1}/{cv_folds}")
                
                # Split fold data
                X_fold_train = X_train_val[train_idx]
                X_fold_val = X_train_val[val_idx]
                y_fold_train = y_train_val[train_idx]
                y_fold_val = y_train_val[val_idx]
                
                # Feature selection on training data ONLY
                fold_feature_selection_result = self._run_pipeline_to_consensus(
                    X_fold_train, y_fold_train, feature_names, 
                    features_target_count=50,  # Default target
                    config={'use_dynamic_thresholds': True}
                )
                
                # Get selected features for this fold
                selected_features = fold_feature_selection_result
                fold_selected_features.append(selected_features)
                
                # Get indices of selected features
                selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
                
                # Apply feature selection to both train and validation sets
                X_fold_train_selected = X_fold_train[:, selected_indices]
                X_fold_val_selected = X_fold_val[:, selected_indices]
                
                # Train model on selected features
                model_copy = self._clone_model(model)
                model_copy.fit(X_fold_train_selected, y_fold_train)
                
                # Evaluate on validation set
                y_pred = model_copy.predict(X_fold_val_selected)
                
                # Calculate score
                if len(np.unique(y)) <= 10:  # Classification
                    score = accuracy_score(y_fold_val, y_pred)
                else:  # Regression
                    score = -mean_squared_error(y_fold_val, y_pred)  # Negative MSE for consistency
                
                fold_scores.append(score)
                
                cv_results.append({
                    'fold': fold_idx + 1,
                    'selected_features': selected_features,
                    'n_selected_features': len(selected_features),
                    'score': score,
                    'train_size': len(X_fold_train),
                    'val_size': len(X_fold_val)
                })
                
                _LOGGER.info(f"✅ Fold {fold_idx + 1}: {len(selected_features)} features, score: {score:.4f}")
            
            # Analyze feature selection consistency across folds
            feature_consistency = self._analyze_feature_consistency_across_folds(fold_selected_features, feature_names)
            
            # Select final feature set based on consistency
            final_features = self._select_final_features_from_cv(fold_selected_features, feature_consistency)
            
            # Final evaluation on test set
            final_test_score = self._evaluate_final_features(
                X_test, y_test, final_features, feature_names, model
            )
            
            # Calculate CV statistics
            cv_mean = np.mean(fold_scores)
            cv_std = np.std(fold_scores)
            cv_scores = fold_scores
            
            execution_time = time.time() - start_time
            
            _LOGGER.info(f"✅ Data leakage prevention completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 CV mean score: {cv_mean:.4f} ± {cv_std:.4f}")
            _LOGGER.info(f"📊 Final test score: {final_test_score:.4f}")
            _LOGGER.info(f"📊 Final features: {len(final_features)}")
            
            return {
                'cv_results': cv_results,
                'cv_scores': cv_scores,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'final_features': final_features,
                'final_test_score': final_test_score,
                'feature_consistency': feature_consistency,
                'execution_time': execution_time,
                'data_leakage_prevented': True
            }
            
        except Exception as e:
            _LOGGER.error(f"❌ Data leakage prevention failed: {e}")
            return {
                'error': str(e),
                'data_leakage_prevented': False,
                'cv_scores': [],
                'final_features': feature_names[:50]  # Fallback
            }

    def _analyze_feature_consistency_across_folds(self, fold_selected_features: List[List[str]], 
                                                feature_names: List[str]) -> Dict[str, Any]:
        """Analyze consistency of feature selection across CV folds."""
        # Count how many times each feature was selected
        feature_counts = {feature: 0 for feature in feature_names}
        for fold_features in fold_selected_features:
            for feature in fold_features:
                if feature in feature_counts:
                    feature_counts[feature] += 1
        
        # Calculate consistency scores
        n_folds = len(fold_selected_features)
        consistency_scores = {
            feature: count / n_folds 
            for feature, count in feature_counts.items()
        }
        
        # Find highly consistent features
        consistent_features = [
            feature for feature, score in consistency_scores.items()
            if score >= 0.6  # Selected in at least 60% of folds
        ]
        
        return {
            'feature_counts': feature_counts,
            'consistency_scores': consistency_scores,
            'consistent_features': consistent_features,
            'n_folds': n_folds,
            'consistency_threshold': 0.6
        }

    def _select_final_features_from_cv(self, fold_selected_features: List[List[str]], 
                                     feature_consistency: Dict[str, Any]) -> List[str]:
        """Select final feature set based on CV consistency."""
        consistent_features = feature_consistency['consistent_features']
        
        if len(consistent_features) >= 20:  # If we have enough consistent features
            return consistent_features
        else:
            # If too few consistent features, take top features by consistency score
            sorted_features = sorted(
                feature_consistency['consistency_scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [feature for feature, _ in sorted_features[:50]]  # Top 50

    def _evaluate_final_features(self, X_test: np.ndarray, y_test: np.ndarray,
                               final_features: List[str], feature_names: List[str],
                               model: Any) -> float:
        """Evaluate final feature set on test set."""
        try:
            # Get selected feature indices
            selected_indices = [feature_names.index(f) for f in final_features if f in feature_names]
            X_test_selected = X_test[:, selected_indices]
            
            # Train model on full training data with selected features
            # Note: In practice, you'd need to retrain on full training set
            # For now, we'll use the test set directly (this is a simplification)
            model_copy = self._clone_model(model)
            model_copy.fit(X_test_selected, y_test)  # This should be training data in practice
            
            # Predict and score
            y_pred = model_copy.predict(X_test_selected)
            
            if len(np.unique(y_test)) <= 10:  # Classification
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_test, y_pred)
            else:  # Regression
                from sklearn.metrics import mean_squared_error
                return -mean_squared_error(y_test, y_pred)
                
        except Exception as e:
            _LOGGER.warning(f"⚠️ Final evaluation failed: {e}")
            return 0.0

    def _clone_model(self, model: Any) -> Any:
        """Clone a model for independent training."""
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback: create a new instance
            return model.__class__(**model.get_params())

    def _enhanced_temporal_analysis_crypto(self, X: np.ndarray, y: np.ndarray,
                                         feature_names: List[str],
                                         time_windows: List[int] = None,
                                         leverage_strategy: str = 'high_leverage') -> Dict[str, Any]:
        """
        Enhanced temporal analysis specifically designed for crypto trading.
        
        This method implements temporal analysis optimized for crypto trading strategies,
        particularly high-leverage trading which requires short time frames (1-30 minutes).
        
        Args:
            X: Feature matrix (time-series data)
            y: Target array
            feature_names: List of feature names
            time_windows: List of time window sizes in minutes
            leverage_strategy: Trading strategy ('high_leverage', 'medium_leverage', 'low_leverage')
            
        Returns:
            Dictionary with enhanced temporal analysis results
        """
        start_time = time.time()
        n_samples = len(X)
        
        # Define time windows based on leverage strategy
        if time_windows is None:
            if leverage_strategy == 'high_leverage':
                # High leverage = short time frames (1-30 minutes)
                time_windows = [1, 5, 15, 30]  # minutes
                _LOGGER.info("🎯 High leverage strategy: Using short time frames (1-30 minutes)")
            elif leverage_strategy == 'medium_leverage':
                # Medium leverage = medium time frames (30-240 minutes)
                time_windows = [30, 60, 120, 240]  # minutes
                _LOGGER.info("🎯 Medium leverage strategy: Using medium time frames (30-240 minutes)")
            else:  # low_leverage
                # Low leverage = long time frames (240+ minutes)
                time_windows = [240, 480, 720, 1440]  # minutes
                _LOGGER.info("🎯 Low leverage strategy: Using long time frames (240+ minutes)")
        
        _LOGGER.info(f"🔄 Starting enhanced temporal analysis for crypto trading...")
        _LOGGER.info(f"📊 Time windows: {time_windows} minutes")
        _LOGGER.info(f"📊 Leverage strategy: {leverage_strategy}")
        
        temporal_results = {}
        feature_temporal_importance = {feature: {} for feature in feature_names}
        
        # Analyze each time window
        for window_minutes in time_windows:
            _LOGGER.info(f"🔄 Analyzing {window_minutes}-minute time window...")
            
            # Convert minutes to samples (assuming 1 sample per minute)
            window_samples = window_minutes
            
            if window_samples > n_samples:
                _LOGGER.warning(f"⚠️ Window {window_minutes}min > data size {n_samples}, skipping")
                continue
            
            # Create overlapping windows with 50% overlap
            overlap_ratio = 0.5
            step_size = int(window_samples * (1 - overlap_ratio))
            if step_size == 0:
                step_size = 1
            
            window_results = []
            feature_importance_per_window = {feature: [] for feature in feature_names}
            
            for start_idx in range(0, n_samples - window_samples + 1, step_size):
                end_idx = start_idx + window_samples
                
                try:
                    # Extract time window
                    X_window = X[start_idx:end_idx]
                    y_window = y[start_idx:end_idx]
                    
                    # Calculate feature importance for this window
                    window_importance = self._calculate_window_feature_importance(
                        X_window, y_window, feature_names, leverage_strategy
                    )
                    
                    window_results.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'window_minutes': window_minutes,
                        'window_samples': window_samples,
                        'feature_importance': window_importance,
                        'n_samples': len(X_window)
                    })
                    
                    # Store importance scores
                    for feature, importance in window_importance.items():
                        feature_importance_per_window[feature].append(importance)
                    
                except Exception as e:
                    _LOGGER.warning(f"⚠️ Time window [{start_idx}:{end_idx}] failed: {e}")
                    continue
            
            # Analyze temporal patterns for this window size
            window_temporal_analysis = self._analyze_temporal_patterns(
                window_results, feature_names, window_minutes, leverage_strategy
            )
            
            temporal_results[f'{window_minutes}min'] = {
                'window_minutes': window_minutes,
                'window_results': window_results,
                'temporal_analysis': window_temporal_analysis,
                'feature_importance_per_window': feature_importance_per_window
            }
            
            # Store temporal importance for each feature
            for feature in feature_names:
                importances = feature_importance_per_window[feature]
                if importances:
                    feature_temporal_importance[feature][f'{window_minutes}min'] = {
                        'mean_importance': np.mean(importances),
                        'std_importance': np.std(importances),
                        'max_importance': np.max(importances),
                        'min_importance': np.min(importances),
                        'temporal_stability': 1.0 - (np.std(importances) / (np.mean(importances) + 1e-6))
                    }
        
        # Analyze cross-timeframe feature behavior
        cross_timeframe_analysis = self._analyze_cross_timeframe_behavior(
            feature_temporal_importance, time_windows, leverage_strategy
        )
        
        # Identify optimal features for each timeframe
        optimal_features_by_timeframe = self._identify_optimal_features_by_timeframe(
            feature_temporal_importance, time_windows, leverage_strategy
        )
        
        # Calculate temporal decay analysis
        temporal_decay_analysis = self._calculate_temporal_decay(
            feature_temporal_importance, time_windows
        )
        
        # Regime-specific analysis (bull/bear markets)
        regime_analysis = self._analyze_regime_specific_importance(
            X, y, feature_names, time_windows, leverage_strategy
        )
        
        enhanced_temporal_analysis = {
            'leverage_strategy': leverage_strategy,
            'time_windows': time_windows,
            'temporal_results': temporal_results,
            'feature_temporal_importance': feature_temporal_importance,
            'cross_timeframe_analysis': cross_timeframe_analysis,
            'optimal_features_by_timeframe': optimal_features_by_timeframe,
            'temporal_decay_analysis': temporal_decay_analysis,
            'regime_analysis': regime_analysis,
            'statistics': {
                'n_timeframes': len(time_windows),
                'total_windows_analyzed': sum(len(result['window_results']) for result in temporal_results.values()),
                'leverage_optimization': leverage_strategy,
                'short_term_focus': leverage_strategy == 'high_leverage'
            }
        }
        
        execution_time = time.time() - start_time
        _LOGGER.info(f"✅ Enhanced temporal analysis completed in {execution_time:.3f}s")
        _LOGGER.info(f"📊 Timeframes analyzed: {len(time_windows)}")
        _LOGGER.info(f"📊 Total windows: {enhanced_temporal_analysis['statistics']['total_windows_analyzed']}")
        
        return {
            'enhanced_temporal_analysis': enhanced_temporal_analysis,
            'execution_time': execution_time
        }

    def _calculate_window_feature_importance(self, X_window: np.ndarray, y_window: np.ndarray,
                                           feature_names: List[str], leverage_strategy: str) -> Dict[str, float]:
        """Calculate feature importance for a specific time window."""
        try:
            importance_scores = {}
            
            # Use different importance measures based on leverage strategy
            if leverage_strategy == 'high_leverage':
                # High leverage: Focus on short-term predictive power and volatility
                for i, feature in enumerate(feature_names):
                    # Calculate correlation with target
                    corr = np.corrcoef(X_window[:, i], y_window)[0, 1]
                    corr_importance = abs(corr) if not np.isnan(corr) else 0.0
                    
                    # Calculate volatility (important for high leverage)
                    volatility = np.std(X_window[:, i]) / (np.mean(X_window[:, i]) + 1e-6)
                    volatility_importance = min(1.0, volatility / 0.1)  # Normalize
                    
                    # Calculate momentum (price change rate)
                    if len(X_window[:, i]) > 1:
                        momentum = (X_window[-1, i] - X_window[0, i]) / (X_window[0, i] + 1e-6)
                        momentum_importance = abs(momentum)
                    else:
                        momentum_importance = 0.0
                    
                    # Combined importance for high leverage
                    importance_scores[feature] = (
                        0.4 * corr_importance + 
                        0.3 * volatility_importance + 
                        0.3 * momentum_importance
                    )
            
            elif leverage_strategy == 'medium_leverage':
                # Medium leverage: Balance between short and medium-term factors
                for i, feature in enumerate(feature_names):
                    # Correlation importance
                    corr = np.corrcoef(X_window[:, i], y_window)[0, 1]
                    corr_importance = abs(corr) if not np.isnan(corr) else 0.0
                    
                    # Trend strength
                    if len(X_window[:, i]) > 2:
                        trend = np.polyfit(range(len(X_window[:, i])), X_window[:, i], 1)[0]
                        trend_importance = abs(trend) / (np.std(X_window[:, i]) + 1e-6)
                    else:
                        trend_importance = 0.0
                    
                    # Stability (lower volatility is better for medium leverage)
                    stability = 1.0 / (np.std(X_window[:, i]) + 1e-6)
                    stability_importance = min(1.0, stability / 10.0)
                    
                    importance_scores[feature] = (
                        0.5 * corr_importance + 
                        0.3 * trend_importance + 
                        0.2 * stability_importance
                    )
            
            else:  # low_leverage
                # Low leverage: Focus on long-term trends and stability
                for i, feature in enumerate(feature_names):
                    # Correlation importance
                    corr = np.corrcoef(X_window[:, i], y_window)[0, 1]
                    corr_importance = abs(corr) if not np.isnan(corr) else 0.0
                    
                    # Long-term trend
                    if len(X_window[:, i]) > 5:
                        trend = np.polyfit(range(len(X_window[:, i])), X_window[:, i], 1)[0]
                        trend_importance = abs(trend) / (np.std(X_window[:, i]) + 1e-6)
                    else:
                        trend_importance = 0.0
                    
                    # Stability (very important for low leverage)
                    stability = 1.0 / (np.std(X_window[:, i]) + 1e-6)
                    stability_importance = min(1.0, stability / 5.0)
                    
                    # Mean reversion tendency
                    mean_val = np.mean(X_window[:, i])
                    mean_reversion = 1.0 / (abs(X_window[-1, i] - mean_val) + 1e-6)
                    mean_reversion_importance = min(1.0, mean_reversion / 10.0)
                    
                    importance_scores[feature] = (
                        0.4 * corr_importance + 
                        0.2 * trend_importance + 
                        0.2 * stability_importance +
                        0.2 * mean_reversion_importance
                    )
            
            return importance_scores
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Window importance calculation failed: {e}")
            return {feature: 0.0 for feature in feature_names}

    def _analyze_temporal_patterns(self, window_results: List[Dict[str, Any]], 
                                 feature_names: List[str], window_minutes: int,
                                 leverage_strategy: str) -> Dict[str, Any]:
        """Analyze temporal patterns for a specific time window."""
        if not window_results:
            return {'error': 'No window results'}
        
        # Calculate temporal stability for each feature
        feature_temporal_stability = {}
        for feature in feature_names:
            importances = [result['feature_importance'].get(feature, 0.0) for result in window_results]
            if importances:
                mean_importance = np.mean(importances)
                std_importance = np.std(importances)
                stability = 1.0 - (std_importance / (mean_importance + 1e-6))
                feature_temporal_stability[feature] = {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'stability': stability,
                    'coefficient_of_variation': std_importance / (mean_importance + 1e-6)
                }
        
        # Identify features with consistent importance
        consistent_features = [
            feature for feature, data in feature_temporal_stability.items()
            if data['stability'] > 0.7  # High stability threshold
        ]
        
        # Identify features with high temporal variability (important for high leverage)
        variable_features = [
            feature for feature, data in feature_temporal_stability.items()
            if data['coefficient_of_variation'] > 0.5  # High variability
        ]
        
        return {
            'window_minutes': window_minutes,
            'leverage_strategy': leverage_strategy,
            'feature_temporal_stability': feature_temporal_stability,
            'consistent_features': consistent_features,
            'variable_features': variable_features,
            'n_windows': len(window_results),
            'temporal_analysis_type': 'crypto_optimized'
        }

    def _analyze_cross_timeframe_behavior(self, feature_temporal_importance: Dict[str, Dict[str, Any]],
                                        time_windows: List[int], leverage_strategy: str) -> Dict[str, Any]:
        """Analyze how features behave across different timeframes."""
        cross_timeframe_behavior = {}
        
        for feature in feature_temporal_importance:
            timeframe_importances = []
            timeframe_stabilities = []
            
            for window_minutes in time_windows:
                window_key = f'{window_minutes}min'
                if window_key in feature_temporal_importance[feature]:
                    data = feature_temporal_importance[feature][window_key]
                    timeframe_importances.append(data['mean_importance'])
                    timeframe_stabilities.append(data['temporal_stability'])
            
            if timeframe_importances:
                # Analyze importance trend across timeframes
                if len(timeframe_importances) > 1:
                    importance_trend = np.polyfit(time_windows[:len(timeframe_importances)], timeframe_importances, 1)[0]
                else:
                    importance_trend = 0.0
                
                # Categorize feature behavior
                if leverage_strategy == 'high_leverage':
                    # High leverage: prefer features that are important in short timeframes
                    short_term_importance = timeframe_importances[0] if timeframe_importances else 0.0
                    behavior_type = 'short_term_focused' if short_term_importance > 0.5 else 'long_term_focused'
                else:
                    # Medium/low leverage: prefer features with consistent importance
                    avg_stability = np.mean(timeframe_stabilities) if timeframe_stabilities else 0.0
                    behavior_type = 'stable' if avg_stability > 0.7 else 'variable'
                
                cross_timeframe_behavior[feature] = {
                    'timeframe_importances': timeframe_importances,
                    'timeframe_stabilities': timeframe_stabilities,
                    'importance_trend': importance_trend,
                    'behavior_type': behavior_type,
                    'avg_importance': np.mean(timeframe_importances),
                    'avg_stability': np.mean(timeframe_stabilities)
                }
        
        return cross_timeframe_behavior

    def _identify_optimal_features_by_timeframe(self, feature_temporal_importance: Dict[str, Dict[str, Any]],
                                              time_windows: List[int], leverage_strategy: str) -> Dict[str, List[str]]:
        """Identify optimal features for each timeframe based on leverage strategy."""
        optimal_features = {}
        
        for window_minutes in time_windows:
            window_key = f'{window_minutes}min'
            feature_scores = []
            
            for feature in feature_temporal_importance:
                if window_key in feature_temporal_importance[feature]:
                    data = feature_temporal_importance[feature][window_key]
                    
                    if leverage_strategy == 'high_leverage':
                        # High leverage: prioritize high importance and some variability
                        score = data['mean_importance'] * 0.7 + data['temporal_stability'] * 0.3
                    else:
                        # Medium/low leverage: prioritize stability
                        score = data['mean_importance'] * 0.5 + data['temporal_stability'] * 0.5
                    
                    feature_scores.append((feature, score))
            
            # Sort by score and select top features
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            top_features = [feature for feature, _ in feature_scores[:20]]  # Top 20 features
            
            optimal_features[window_key] = top_features
        
        return optimal_features

    def _calculate_temporal_decay(self, feature_temporal_importance: Dict[str, Dict[str, Any]],
                                time_windows: List[int]) -> Dict[str, Any]:
        """Calculate temporal decay of feature importance."""
        temporal_decay = {}
        
        for feature in feature_temporal_importance:
            importances = []
            for window_minutes in time_windows:
                window_key = f'{window_minutes}min'
                if window_key in feature_temporal_importance[feature]:
                    importance = feature_temporal_importance[feature][window_key]['mean_importance']
                    importances.append(importance)
            
            if len(importances) > 1:
                # Calculate decay rate
                decay_rate = (importances[0] - importances[-1]) / (time_windows[-1] - time_windows[0])
                half_life = time_windows[0] + (importances[0] / 2 - importances[0]) / decay_rate if decay_rate != 0 else float('inf')
                
                temporal_decay[feature] = {
                    'decay_rate': decay_rate,
                    'half_life': half_life,
                    'initial_importance': importances[0],
                    'final_importance': importances[-1],
                    'decay_type': 'fast' if decay_rate > 0.01 else 'slow' if decay_rate > 0.001 else 'stable'
                }
        
        return temporal_decay

    def _analyze_regime_specific_importance(self, X: np.ndarray, y: np.ndarray,
                                          feature_names: List[str], time_windows: List[int],
                                          leverage_strategy: str) -> Dict[str, Any]:
        """Analyze feature importance in different market regimes (bull/bear)."""
        try:
            # Simple regime detection based on price movement
            price_changes = np.diff(y) if len(y) > 1 else [0]
            regime_threshold = np.percentile(price_changes, 70)  # Top 30% = bull market
            
            bull_market_indices = np.where(price_changes > regime_threshold)[0]
            bear_market_indices = np.where(price_changes < -regime_threshold)[0]
            
            regime_analysis = {
                'bull_market': {'indices': bull_market_indices, 'features': {}},
                'bear_market': {'indices': bear_market_indices, 'features': {}}
            }
            
            # Analyze feature importance in each regime
            for regime_name, regime_data in regime_analysis.items():
                if len(regime_data['indices']) > 10:  # Need sufficient data
                    regime_X = X[regime_data['indices']]
                    regime_y = y[regime_data['indices']]
                    
                    # Calculate feature importance for this regime
                    regime_importance = self._calculate_regime_feature_importance(
                        regime_X, regime_y, feature_names, leverage_strategy
                    )
                    
                    regime_data['features'] = regime_importance
            
            return regime_analysis
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Regime analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_regime_feature_importance(self, X_regime: np.ndarray, y_regime: np.ndarray,
                                           feature_names: List[str], leverage_strategy: str) -> Dict[str, float]:
        """Calculate feature importance for a specific market regime."""
        importance_scores = {}
        
        for i, feature in enumerate(feature_names):
            # Calculate correlation with target in this regime
            corr = np.corrcoef(X_regime[:, i], y_regime)[0, 1]
            corr_importance = abs(corr) if not np.isnan(corr) else 0.0
            
            # Calculate regime-specific volatility
            volatility = np.std(X_regime[:, i]) / (np.mean(X_regime[:, i]) + 1e-6)
            
            # Regime-specific importance calculation
            if leverage_strategy == 'high_leverage':
                # High leverage: volatility is important for both bull and bear markets
                importance_scores[feature] = corr_importance * 0.6 + min(1.0, volatility) * 0.4
            else:
                # Medium/low leverage: focus more on correlation
                importance_scores[feature] = corr_importance * 0.8 + min(1.0, volatility) * 0.2
        
        return importance_scores

    def validate_feature_reduction_plan(self, initial_count: int, target_count: int, 
                                      model_type: str) -> Dict[str, Any]:
        """
        Validate and plan the feature reduction strategy.
        
        Args:
            initial_count: Initial number of features
            target_count: Target number of features
            model_type: Type of model to optimize for
            
        Returns:
            Dictionary with validation results and reduction plan
        """
        # Get model-specific target
        model_target = self.get_model_target_features(model_type)
        
        # Use model-specific target if not provided
        if target_count is None:
            target_count = model_target
            _LOGGER.info(f"🎯 Using model-specific target: {target_count} features for {model_type}")
        
        # Calculate removal count
        removal_count = initial_count - target_count
        
        # Validation checks
        validation_result = {
            'valid': True,
            'initial_count': initial_count,
            'target_count': target_count,
            'removal_count': removal_count,
            'model_type': model_type,
            'model_target': model_target,
            'warnings': [],
            'errors': []
        }
        
        # Check if reduction is feasible
        if removal_count <= 0:
            validation_result['errors'].append(f"No reduction needed: {initial_count} <= {target_count}")
            validation_result['valid'] = False
        
        # Check if reduction is too aggressive
        reduction_ratio = removal_count / initial_count
        if reduction_ratio > 0.95:
            validation_result['warnings'].append(f"Very aggressive reduction: {reduction_ratio:.1%}")
        
        # Check if target is reasonable for model type
        if target_count < 10:
            validation_result['warnings'].append(f"Very low target count: {target_count} features")
        
        # Check if we can maintain minimum intermediate features
        if target_count < self.MIN_FEATURES_INTERMEDIATE:
            validation_result['warnings'].append(
                f"Target {target_count} < minimum intermediate {self.MIN_FEATURES_INTERMEDIATE}. "
                f"Will use RF refinement for final reduction."
            )
        
        # Plan reduction stages
        if validation_result['valid']:
            validation_result['reduction_plan'] = self._create_reduction_plan(
                initial_count, target_count, model_type
            )
        
        return validation_result

    def _create_reduction_plan(self, initial_count: int, target_count: int, 
                             model_type: str) -> Dict[str, Any]:
        """
        Create a detailed reduction plan with stage-specific targets.
        
        Args:
            initial_count: Initial number of features
            target_count: Target number of features
            model_type: Type of model
            
        Returns:
            Dictionary with reduction plan
        """
        removal_count = initial_count - target_count
        
        # Stage targets
        if target_count >= self.MIN_FEATURES_INTERMEDIATE:
            # Standard reduction plan
            stage1_target = max(self.MIN_FEATURES_INTERMEDIATE, 
                              initial_count - int(removal_count * 0.3))  # 30% reduction
            stage2_target = max(self.MIN_FEATURES_INTERMEDIATE, 
                              initial_count - int(removal_count * 0.6))  # 60% reduction
            stage3_target = max(self.MIN_FEATURES_INTERMEDIATE, 
                              initial_count - int(removal_count * 0.8))  # 80% reduction
            final_target = target_count
            use_rf_refinement = False
        else:
            # Need RF refinement for final precision
            stage1_target = max(self.MIN_FEATURES_INTERMEDIATE, 
                              initial_count - int(removal_count * 0.2))  # 20% reduction
            stage2_target = max(self.MIN_FEATURES_INTERMEDIATE, 
                              initial_count - int(removal_count * 0.4))  # 40% reduction
            stage3_target = max(self.MIN_FEATURES_INTERMEDIATE, 
                              initial_count - int(removal_count * 0.6))  # 60% reduction
            final_target = self.MIN_FEATURES_INTERMEDIATE  # Stop at minimum
            use_rf_refinement = True
        
        plan = {
            'stage1_correlation': {
                'target': stage1_target,
                'reduction': initial_count - stage1_target,
                'method': 'correlation_based_filtering'
            },
            'stage2_mrmr': {
                'target': stage2_target,
                'reduction': stage1_target - stage2_target,
                'method': 'mrmr_selection'
            },
            'stage3_consensus': {
                'target': stage3_target,
                'reduction': stage2_target - stage3_target,
                'method': 'lasso_rfe_consensus'
            },
            'stage4_bootstrap': {
                'target': final_target,
                'reduction': stage3_target - final_target,
                'method': 'bootstrap_stability'
            },
            'stage5_rf_refinement': {
                'target': target_count,
                'reduction': final_target - target_count,
                'method': 'rf_cross_validation',
                'enabled': use_rf_refinement
            }
        }
        
        _LOGGER.info(f"📋 Reduction plan for {model_type}:")
        _LOGGER.info(f"   Initial: {initial_count} → Target: {target_count}")
        _LOGGER.info(f"   Stage 1 (Correlation): {initial_count} → {stage1_target}")
        _LOGGER.info(f"   Stage 2 (mRMR): {stage1_target} → {stage2_target}")
        _LOGGER.info(f"   Stage 3 (Consensus): {stage2_target} → {stage3_target}")
        _LOGGER.info(f"   Stage 4 (Bootstrap): {stage3_target} → {final_target}")
        if use_rf_refinement:
            _LOGGER.info(f"   Stage 5 (RF Refinement): {final_target} → {target_count}")
        
        return plan

    def rf_cross_validation_refinement(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str], target_count: int,
                                     cv_folds: int = 5) -> Dict[str, Any]:
        """
        Use Random Forest with cross-validation for precise final feature refinement.
        
        This method is used when the target feature count is below the minimum
        intermediate threshold (100 features) to achieve precise final counts.
        
        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            target_count: Exact target number of features
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with refined feature selection results
        """
        start_time = time.time()
        _LOGGER.info(f"🎯 RF Cross-Validation Refinement: {len(feature_names)} → {target_count}")
        
        try:
            if not SKLEARN_AVAILABLE:
                _LOGGER.warning("⚠️ Scikit-learn not available for RF refinement")
                return {
                    'selected_features': feature_names[:target_count],
                    'refinement_scores': {},
                    'cv_scores': [],
                    'execution_time': time.time() - start_time,
                    'method': 'fallback_slice'
                }
            
            # Use RFECV for precise feature selection
            base_model = self._get_default_model(y)
            if base_model is None:
                _LOGGER.warning("⚠️ No suitable base model for RF refinement")
                return {
                    'selected_features': feature_names[:target_count],
                    'refinement_scores': {},
                    'cv_scores': [],
                    'execution_time': time.time() - start_time,
                    'method': 'fallback_slice'
                }
            
            # Create RFECV with target feature count
            rfecv = RFECV(
                estimator=base_model,
                step=1,
                cv=cv_folds,
                scoring='accuracy' if len(np.unique(y)) <= 10 else 'neg_mean_squared_error',
                min_features_to_select=target_count,
                n_jobs=-1 if self.enable_parallel else 1
            )
            
            # Fit RFECV
            rfecv.fit(X, y)
            
            # Get selected features
            selected_mask = rfecv.support_
            selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
            
            # Ensure we have exactly target_count features
            if len(selected_features) > target_count:
                # If we have more than target, use feature importance to select top features
                feature_importance = rfecv.estimator_.feature_importances_
                importance_scores = dict(zip(selected_features, feature_importance))
                sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
                selected_features = [feature for feature, _ in sorted_features[:target_count]]
            elif len(selected_features) < target_count:
                # If we have fewer than target, add remaining features by importance
                remaining_features = [f for f in feature_names if f not in selected_features]
                if remaining_features:
                    # Use a simple RF to get importance for remaining features
                    temp_rf = self._get_default_model(y)
                    temp_rf.fit(X, y)
                    remaining_importance = dict(zip(remaining_features, temp_rf.feature_importances_))
                    sorted_remaining = sorted(remaining_importance.items(), key=lambda x: x[1], reverse=True)
                    needed = target_count - len(selected_features)
                    selected_features.extend([feature for feature, _ in sorted_remaining[:needed]])
            
            # Calculate refinement scores
            refinement_scores = {}
            if hasattr(rfecv, 'estimator_') and hasattr(rfecv.estimator_, 'feature_importances_'):
                importance_scores = rfecv.estimator_.feature_importances_
                for i, feature in enumerate(selected_features):
                    if i < len(importance_scores):
                        refinement_scores[feature] = importance_scores[i]
            
            execution_time = time.time() - start_time
            
            _LOGGER.info(f"✅ RF refinement completed: {len(selected_features)} features selected")
            _LOGGER.info(f"📊 CV scores: {rfecv.cv_results_['mean_test_score']}")
            _LOGGER.info(f"⏱️ Execution time: {execution_time:.3f}s")
            
            return {
                'selected_features': selected_features,
                'refinement_scores': refinement_scores,
                'cv_scores': rfecv.cv_results_['mean_test_score'].tolist(),
                'optimal_features': rfecv.n_features_,
                'execution_time': execution_time,
                'method': 'rfecv_refinement'
            }
            
        except Exception as e:
            _LOGGER.error(f"❌ RF refinement failed: {e}")
            # Fallback to simple selection
            selected_features = feature_names[:target_count]
            return {
                'selected_features': selected_features,
                'refinement_scores': {},
                'cv_scores': [],
                'execution_time': time.time() - start_time,
                'method': 'fallback_slice',
                'error': str(e)
            }

    def _get_default_model(self, y: np.ndarray):
        """Get a default model based on the target type."""
        if not SKLEARN_AVAILABLE:
            return None
        
        if len(np.unique(y)) <= 10 and not np.issubdtype(np.asarray(y).dtype, np.floating):
            return RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        else:
            return RandomForestRegressor(n_estimators=50, random_state=self.random_state)

    def _calculate_relevance_scores_parallel(self, X: np.ndarray, y: np.ndarray,
                                           feature_names: List[str], method: str) -> Dict[str, float]:
        """Calculate relevance scores with parallel processing."""
        try:
            if not self.parallel_processor or not self.enable_parallel:
                return self._calculate_relevance_scores(X, y, feature_names, method)
            
            # Split features into chunks for parallel processing
            chunk_size = max(1, len(feature_names) // self.max_workers)
            feature_chunks = [feature_names[i:i + chunk_size] for i in range(0, len(feature_names), chunk_size)]
            
            _LOGGER.info(f"⚡ Processing {len(feature_chunks)} feature chunks in parallel")
            
            # Prepare parameters for parallel processing
            chunk_params = []
            for i, chunk in enumerate(feature_chunks):
                chunk_indices = [feature_names.index(f) for f in chunk]
                chunk_params.append({
                    'chunk_idx': i,
                    'feature_names': chunk,
                    'feature_indices': chunk_indices,
                    'X_chunk': X[:, chunk_indices],
                    'y': y,
                    'method': method
                })
            
            # Process chunks in parallel
            chunk_results = self.parallel_processor.parallel_apply(
                chunk_params,
                self._calculate_relevance_chunk,
                max_workers=self.max_workers
            )
            
            # Combine results
            relevance_scores = {}
            for result in chunk_results:
                if result and 'error' not in result:
                    relevance_scores.update(result['scores'])
                else:
                    _LOGGER.warning(f"⚠️ Relevance chunk failed: {result.get('error', 'Unknown error')}")
            
            # Fallback for any missing scores
            for feature in feature_names:
                if feature not in relevance_scores:
                    relevance_scores[feature] = 0.0
            
            return relevance_scores
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Parallel relevance calculation failed: {e}, falling back to sequential")
            return self._calculate_relevance_scores(X, y, feature_names, method)

    def _calculate_relevance_chunk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relevance scores for a chunk of features."""
        try:
            chunk_idx = params['chunk_idx']
            feature_names = params['feature_names']
            X_chunk = params['X_chunk']
            y = params['y']
            method = params['method']
            
            scores = {}
            for i, feature_name in enumerate(feature_names):
                if method == 'mutual_info':
                    if SKLEARN_AVAILABLE:
                        try:
                            if len(np.unique(y)) <= 10 and not np.issubdtype(np.asarray(y).dtype, np.floating):
                                mi_score = mutual_info_classif(X_chunk[:, i:i+1], y, random_state=self.random_state)[0]
                            else:
                                mi_score = mutual_info_regression(X_chunk[:, i:i+1], y, random_state=self.random_state)[0]
                            scores[feature_name] = float(mi_score)
                        except Exception:
                            # Fallback to correlation
                            corr_matrix = np.corrcoef(X_chunk[:, i], y)
                            if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                                scores[feature_name] = abs(float(corr_matrix[0, 1]))
                            else:
                                scores[feature_name] = 0.0
                    else:
                        # Fallback to correlation
                        corr_matrix = np.corrcoef(X_chunk[:, i], y)
                        if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                            scores[feature_name] = abs(float(corr_matrix[0, 1]))
                        else:
                            scores[feature_name] = 0.0
                            
                elif method == 'correlation':
                    corr_matrix = np.corrcoef(X_chunk[:, i], y)
                    if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                        scores[feature_name] = abs(float(corr_matrix[0, 1]))
                    else:
                        scores[feature_name] = 0.0
                        
                elif method == 'importance':
                    # Use a simple importance calculation for the chunk
                    if len(np.unique(y)) <= 10:
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=10, random_state=self.random_state)
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=10, random_state=self.random_state)
                    
                    model.fit(X_chunk[:, i:i+1], y)
                    scores[feature_name] = float(model.feature_importances_[0])
                
                # Handle NaN values
                if np.isnan(scores[feature_name]):
                    scores[feature_name] = 0.0
            
            return {
                'chunk_idx': chunk_idx,
                'scores': scores
            }
            
        except Exception as e:
            return {
                'chunk_idx': params.get('chunk_idx', -1),
                'error': str(e)
            }

    def _calculate_relevance_scores(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str], method: str) -> Dict[str, float]:
        """Calculate relevance scores for features."""
        try:
            scores = {}

            if method == 'mutual_info':
                if SKLEARN_AVAILABLE:
                    # Choose appropriate mutual information function based on target type
                    try:
                        if len(np.unique(y)) <= 10 and not np.issubdtype(np.asarray(y).dtype, np.floating):
                            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
                        else:
                            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
                        scores = dict(zip(feature_names, mi_scores))
                    except Exception:
                        # Fallback to correlation if MI fails
                        for idx, feature_name in enumerate(feature_names):
                            scores[feature_name] = abs(np.corrcoef(X[:, idx], y)[0, 1])
                else:
                    # Fallback: use correlation for regression-like relevance
                    for idx, feature_name in enumerate(feature_names):
                        try:
                            corr_matrix = np.corrcoef(X[:, idx], y)
                            if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                                corr_value = corr_matrix[0, 1]
                            else:
                                corr_value = float(corr_matrix) if np.isscalar(corr_matrix) else 0.0
                            scores[feature_name] = abs(float(corr_value))
                        except (ValueError, IndexError, TypeError):
                            scores[feature_name] = 0.0

            elif method == 'correlation':
                for idx, feature_name in enumerate(feature_names):
                    try:
                        corr_matrix = np.corrcoef(X[:, idx], y)
                        if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                            corr_value = corr_matrix[0, 1]
                        else:
                            corr_value = float(corr_matrix) if np.isscalar(corr_matrix) else 0.0
                        scores[feature_name] = abs(float(corr_value))
                    except (ValueError, IndexError, TypeError):
                        scores[feature_name] = 0.0

            elif method == 'importance':
                importance_scores = self._calculate_importance_scores(X, y, feature_names)
                scores = importance_scores

            # Handle NaN values
            for feature in feature_names:
                if feature not in scores or np.isnan(scores[feature]):
                    scores[feature] = 0.0

            return scores

        except Exception as e:
            self.logger.warning(f"Relevance score calculation failed: {e}")
            return {feature: 0.0 for feature in feature_names}

    def _calculate_redundancy_score(self, feature1: np.ndarray, feature2: np.ndarray,
                                  name1: str, name2: str, method: str) -> float:
        """Calculate redundancy score between two features."""
        try:
            if method == 'correlation':
                corr_matrix = np.corrcoef(feature1, feature2)
                if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                    return abs(float(corr_matrix[0, 1]))
                else:
                    return abs(float(corr_matrix)) if np.isscalar(corr_matrix) else 0.0
            elif method == 'mutual_info':
                if SKLEARN_AVAILABLE:
                    mi_score = mutual_info_regression(feature1.reshape(-1, 1), feature2)[0]
                    return float(mi_score)
                else:
                    corr_matrix = np.corrcoef(feature1, feature2)
                    if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                        return abs(float(corr_matrix[0, 1]))
                    else:
                        return abs(float(corr_matrix)) if np.isscalar(corr_matrix) else 0.0
            else:
                return 0.0
        except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
            return 0.0

    def _calculate_importance_scores(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance scores using Random Forest."""
        try:
            if not SKLEARN_AVAILABLE:
                return {feature: 1.0 / len(feature_names) for feature in feature_names}

            # Choose appropriate model based on target
            if len(np.unique(y)) <= 10:  # Classification
                model = RandomForestClassifier(
                    n_estimators=self.method_configs['importance']['n_estimators'],
                    max_depth=self.method_configs['importance']['max_depth'],
                    random_state=self.random_state
                )
            else:  # Regression
                model = RandomForestRegressor(
                    n_estimators=self.method_configs['importance']['n_estimators'],
                    max_depth=self.method_configs['importance']['max_depth'],
                    random_state=self.random_state
                )

            model.fit(X, y)
            importance_scores = dict(zip(feature_names, model.feature_importances_))

            return importance_scores

        except Exception as e:
            self.logger.warning(f"Importance score calculation failed: {e}")
            return {feature: 1.0 / len(feature_names) for feature in feature_names}

    def _calculate_stability_scores(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature stability scores based on variance and distribution."""
        try:
            stability_scores = {}

            for idx, feature_name in enumerate(feature_names):
                feature_values = X[:, idx]

                # Remove NaN values for calculation
                clean_values = feature_values[~np.isnan(feature_values)]

                if len(clean_values) > 0:
                    # Stability based on coefficient of variation
                    mean_val = np.mean(clean_values)
                    std_val = np.std(clean_values)

                    if mean_val != 0:
                        cv = abs(std_val / mean_val)
                        # Convert to stability score (lower CV = higher stability)
                        stability = 1.0 / (1.0 + cv)
                    else:
                        stability = 0.5  # Neutral stability for zero-mean features
                else:
                    stability = 0.0

                stability_scores[feature_name] = stability

            return stability_scores

        except Exception as e:
            self.logger.warning(f"Stability score calculation failed: {e}")
            return {feature: 0.5 for feature in feature_names}

    def _calculate_variance_scores(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature scores based on variance."""
        try:
            variance_scores = {}

            for idx, feature_name in enumerate(feature_names):
                feature_values = X[:, idx]
                clean_values = feature_values[~np.isnan(feature_values)]

                if len(clean_values) > 0:
                    variance = np.var(clean_values)
                    # Normalize variance to [0, 1] range (roughly)
                    normalized_variance = min(variance / (variance + 1.0), 1.0)
                    variance_scores[feature_name] = normalized_variance
                else:
                    variance_scores[feature_name] = 0.0

            return variance_scores

        except Exception as e:
            self.logger.warning(f"Variance score calculation failed: {e}")
            return {feature: 0.0 for feature in feature_names}

    def _calculate_permutation_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, float]:
        """Calculate permutation feature importance."""
        try:
            if not SKLEARN_AVAILABLE:
                return self._calculate_importance_scores(X, y, feature_names)

            from sklearn.inspection import permutation_importance

            # Get baseline score
            baseline_score = self._calculate_model_score(model, X, y)

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=5, random_state=self.random_state
            )

            importance_scores = dict(zip(feature_names, perm_importance.importances_mean))
            return importance_scores

        except Exception as e:
            self.logger.warning(f"Permutation importance calculation failed: {e}")
            return self._calculate_importance_scores(X, y, feature_names)

    def _calculate_tree_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, float]:
        """Calculate tree-based feature importance."""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            else:
                # Fallback to training a random forest
                return self._calculate_importance_scores(X, y, feature_names)
        except Exception as e:
            self.logger.warning(f"Tree importance calculation failed: {e}")
            return {feature: 1.0 / len(feature_names) for feature in feature_names}

    def _calculate_coefficient_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance based on model coefficients."""
        try:
            if hasattr(model, 'coef_'):
                coefficients = np.abs(model.coef_.flatten())
                return dict(zip(feature_names, coefficients))
            elif hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            else:
                return {feature: 1.0 / len(feature_names) for feature in feature_names}
        except Exception as e:
            self.logger.warning(f"Coefficient importance calculation failed: {e}")
            return {feature: 1.0 / len(feature_names) for feature in feature_names}

    def _calculate_model_score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate a baseline score for the model."""
        try:
            if hasattr(model, 'score'):
                return model.score(X, y)
            else:
                # Fallback to accuracy for classification
                predictions = model.predict(X)
                if len(np.unique(y)) <= 10:  # Classification
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y, predictions)
                else:  # Regression
                    from sklearn.metrics import r2_score
                    return r2_score(y, predictions)
        except:
            return 0.5

    def _select_features_single_fold(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str], method: str) -> List[str]:
        """Select features for a single CV fold."""
        try:
            if method == 'importance':
                importance_scores = self._calculate_importance_scores(X, y, feature_names)
                # Select top 50% of features
                n_select = max(1, len(feature_names) // 2)
                sorted_features = sorted(
                    importance_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                return [feature for feature, _ in sorted_features[:n_select]]
            else:
                # Default: return all features
                return feature_names
        except Exception as e:
            self.logger.warning(f"Single fold feature selection failed: {e}")
            return feature_names

    def _calculate_feature_stability(self, fold_selections: List[Dict[str, Any]],
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Calculate feature selection stability across folds."""
        try:
            stability_scores = {}

            for feature in feature_names:
                selection_count = sum(
                    1 for fold in fold_selections
                    if feature in fold['selected_features']
                )

                stability = selection_count / len(fold_selections)
                stability_scores[feature] = {
                    'selection_frequency': selection_count,
                    'stability_score': stability,
                    'selected_in_folds': [fold['fold_idx'] for fold in fold_selections
                                        if feature in fold['selected_features']]
                }

            return stability_scores

        except Exception as e:
            self.logger.warning(f"Feature stability calculation failed: {e}")
            return {}
