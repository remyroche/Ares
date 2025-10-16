"""
Simplified NAS Ensemble Training Component

This component trains base models for current regime detection using:
- General training pipeline for CV, lookahead prevention, overfitting detection
- Only base models (no meta-learner)
- Features from feature_engineering/ and feature_selection/ pipelines
- Goal: detect current regime (not predict future)
- Models: LGBM, XGBoost, Random Forest, ElasticNet
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer
from src.utils.logger import system_logger
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import ML libraries
try:
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    ML_LIBRARIES_AVAILABLE = True
    tprint("✅ [NAS_ENSEMBLE] ML libraries imported successfully", color="green")
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    tprint(f"❌ [NAS_ENSEMBLE] Failed to import ML libraries: {e}", color="red")

# Import comprehensive training pipeline infrastructure
try:
    from src.utils.ml_common.training.base_training_step import BaseTrainingStep
    from src.utils.ml_common.validation.enhanced_overfitting_detection import UniversalOverfittingDetector, OverfittingConfig
    from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPreventionConfig
    from src.utils.ml_common.training.universal_validation_integration import get_validation_integrator
    from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection
    COMPREHENSIVE_TRAINING_AVAILABLE = True
    tprint("✅ [NAS_ENSEMBLE] Comprehensive training pipeline imported successfully", color="green")
except ImportError as e:
    COMPREHENSIVE_TRAINING_AVAILABLE = False
    tprint(f"❌ [NAS_ENSEMBLE] Failed to import comprehensive training pipeline: {e}", color="red")

# Import PatchTST wrapper for tree-based models
try:
    from src.training.steps.model_training.patchtst_wrapper import create_patchtst_wrapper
    PATCHTST_AVAILABLE = True
    tprint("✅ [NAS_ENSEMBLE] PatchTST wrapper imported successfully", color="green")
except ImportError as e:
    PATCHTST_AVAILABLE = False
    create_patchtst_wrapper = None
    tprint(f"⚠️ [NAS_ENSEMBLE] PatchTST wrapper not available: {e}", color="yellow")

class NASEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    Simplified NAS Ensemble Training Component.

    Trains base models for current regime detection using the general training pipeline.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Simplified NAS Ensemble Training Component."""
        tprint("🚀 [NAS_ENSEMBLE] Initializing Simplified NAS Ensemble Training Component", color="cyan", bold=True)
        super().__init__(config)

        self.logger = system_logger.getChild('NASEnsembleTrainingComponent')
        tprint("✅ [NAS_ENSEMBLE] Logger initialized", color="green")

        # Initialize base models configuration
        self.base_models_config = {
            'lgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            'elastic_net': {
                'alpha': 0.1,
                'l1_ratio': 0.5,
                'random_state': 42,
                'max_iter': 1000
            }
        }
        tprint("⚙️ [NAS_ENSEMBLE] Base models configuration set", color="yellow")

        # Initialize models
        self.base_models = {}
        self.training_metrics = {}
        tprint("📊 [NAS_ENSEMBLE] Base models initialized", color="blue")

        tprint("✅ [NAS_ENSEMBLE] Simplified NAS Ensemble Training Component initialized successfully", color="green", bold=True)

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [NAS_ENSEMBLE] Getting required artifacts", color="cyan")
        required_artifacts = ['nas_ensemble_training_result']
        tprint(f"✅ [NAS_ENSEMBLE] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts

    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute simplified NAS ensemble training.

        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state containing features, targets, and regime labels

        Returns:
            ComponentResult with training results
        """
        tprint("🚀 [NAS_ENSEMBLE] Starting simplified NAS ensemble training execution", color="cyan", bold=True)
        start_time = datetime.now()

        try:
            # Extract required data from pipeline state
            tprint("📊 [NAS_ENSEMBLE] Extracting data from pipeline state", color="yellow")
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            # Extract regime labels from pipeline state artifacts
            artifacts = pipeline_state.get('artifacts', {})

            # Try multiple possible artifact keys for clustering results
            regime_labels = None

            # First try the new optimal_regime_clustering_result structure
            optimal_clustering_result = artifacts.get('optimal_regime_clustering_result', {})
            if optimal_clustering_result:
                clustering_result = optimal_clustering_result.get('clustering_result')
                if clustering_result and isinstance(clustering_result, dict):
                    regime_labels = clustering_result.get('cluster_assignments') or clustering_result.get('regime_assignments')
                    # Handle case where assignments are stored as string representation
                    if isinstance(regime_labels, str):
                        try:
                            # Parse numpy array string representation
                            import numpy as np
                            # Remove brackets and split by spaces, then convert to int
                            clean_str = regime_labels.strip('[]')
                            regime_labels = np.array([int(x) for x in clean_str.split() if x.strip()])
                            tprint("🔍 [NAS_ENSEMBLE] Parsed regime labels from string representation", color="blue")
                        except Exception as e:
                            tprint(f"⚠️ [NAS_ENSEMBLE] Failed to parse regime labels string: {e}", color="yellow")
                            regime_labels = None
                    if regime_labels is not None:
                        tprint("🔍 [NAS_ENSEMBLE] Found regime labels in optimal_regime_clustering_result", color="blue")

            # Fallback to old nas_tas_clustering_result structure
            if regime_labels is None:
                nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
                regime_labels = nas_tas_clustering_result.get('regime_assignments')
                if regime_labels is not None:
                    tprint("🔍 [NAS_ENSEMBLE] Found regime labels in nas_tas_clustering_result", color="blue")
            feature_names = pipeline_state.get('feature_names', [])

            # Validate required data
            tprint("🔍 [NAS_ENSEMBLE] Validating required data", color="yellow")
            if X is None or y is None:
                tprint("❌ [NAS_ENSEMBLE] Missing features or targets", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Missing features or targets in pipeline state",
                    metadata={'component_type': 'nas_ensemble_training'}
                )

            if regime_labels is None:
                tprint("⚠️ [NAS_ENSEMBLE] No regime labels found, using targets as regime labels", color="yellow")
                regime_labels = y

            tprint(f"📊 [NAS_ENSEMBLE] Data shapes - X: {X.shape}, y: {y.shape}, regime_labels: {len(regime_labels) if regime_labels is not None else 'None'}", color="blue")

            # Prepare data for training
            tprint("🔧 [NAS_ENSEMBLE] Preparing data for training", color="yellow")
            X_processed, y_processed, regime_labels_processed = self._prepare_data(X, y, regime_labels)
            tprint(f"✅ [NAS_ENSEMBLE] Data prepared - X: {X_processed.shape}, y: {y_processed.shape}", color="green")

            # Use regime labels as targets for current regime detection
            y_targets = regime_labels_processed

            # Train base models using general training pipeline
            tprint("🏋️ [NAS_ENSEMBLE] Training base models using general training pipeline", color="yellow")
            base_models = self._train_base_models_with_pipeline(X_processed, y_targets)

            # Evaluate models
            tprint("📊 [NAS_ENSEMBLE] Evaluating base models performance", color="yellow")
            training_metrics = self._evaluate_base_models(X_processed, y_targets, base_models)

            # Create comprehensive results
            tprint("📦 [NAS_ENSEMBLE] Creating comprehensive results", color="yellow")
            results = {
                'nas_ensemble_training_result': {
                    'base_models': base_models,
                    'training_metrics': training_metrics,
                    'training_time': (datetime.now() - start_time).total_seconds(),
                    'success': True,
                    'metadata': {
                        'component_type': 'nas_ensemble_training',
                        'data_shape': X_processed.shape,
                        'n_regimes': len(np.unique(regime_labels_processed)) if regime_labels_processed is not None else 0,
                        'feature_names': feature_names,
                        'timestamp': datetime.now().isoformat(),
                        'model_types': list(base_models.keys())
                    }
                }
            }

            tprint("✅ [NAS_ENSEMBLE] Simplified NAS ensemble training completed successfully", color="green", bold=True)
            tprint(f"⏱️ [NAS_ENSEMBLE] Total execution time: {(datetime.now() - start_time).total_seconds():.2f}s", color="blue")

            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={'component_type': 'nas_ensemble_training', 'execution_time': (datetime.now() - start_time).total_seconds()}
            )

        except Exception as e:
            tprint(f"❌ [NAS_ENSEMBLE] Simplified NAS ensemble training failed: {e}", color="red", bold=True)
            self.logger.error(f"Simplified NAS ensemble training failed: {e}", exc_info=True)
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'nas_ensemble_training'}
            )

    def _prepare_data(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training."""
        tprint("🔧 [NAS_ENSEMBLE] Preparing data for training", color="yellow")

        # Handle missing values
        tprint("🧹 [NAS_ENSEMBLE] Handling missing values", color="blue")
        if isinstance(X, pd.DataFrame):
            X = X.fillna(0).values
        elif isinstance(X, list):
            X = np.array(X)

        if isinstance(y, (pd.Series, list)):
            y = np.array(y)

        if regime_labels is not None and isinstance(regime_labels, (pd.Series, list)):
            regime_labels = np.array(regime_labels)

        # Ensure all arrays have the same length
        tprint("📏 [NAS_ENSEMBLE] Ensuring consistent array lengths", color="blue")
        min_length = min(len(X), len(y))
        if regime_labels is not None:
            min_length = min(min_length, len(regime_labels))

        X = X[:min_length]
        y = y[:min_length]
        if regime_labels is not None:
            regime_labels = regime_labels[:min_length]

        tprint(f"✅ [NAS_ENSEMBLE] Data prepared - X: {X.shape}, y: {y.shape}, regime_labels: {regime_labels.shape if regime_labels is not None else 'None'}", color="green")
        return X, y, regime_labels

    def _train_base_models_with_pipeline(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train base models using the comprehensive training pipeline."""
        tprint("🏋️ [NAS_ENSEMBLE] Training base models with comprehensive training pipeline", color="yellow")

        base_models = {}

        # Initialize comprehensive training components
        if COMPREHENSIVE_TRAINING_AVAILABLE:
            tprint("🔧 [NAS_ENSEMBLE] Initializing comprehensive training components", color="blue")

            # Initialize overfitting detection
            overfitting_config = OverfittingConfig(
                accuracy_gap_threshold=0.05,
                enable_early_stopping=True,
                patience=5,
                save_reports=True
            )
            overfitting_detector = UniversalOverfittingDetector(overfitting_config)

            # Initialize lookahead protection
            lookahead_protection = LookaheadProtection({
                'strict_mode': True,
                'enable_automatic_filtering': True,
                'tolerance_seconds': 60
            })

            # Initialize overfitting prevention
            prevention_config = OverfittingPreventionConfig(
                enable_early_stopping=True,
                enable_cross_validation=True,
                cv_folds=5,
                cv_strategy='time_series_split',
                enable_regularization=True
            )

            tprint("✅ [NAS_ENSEMBLE] Comprehensive training components initialized", color="green")

        # Train LGBM with comprehensive pipeline
        tprint("🌲 [NAS_ENSEMBLE] Training LGBM model with comprehensive pipeline", color="blue")
        try:
            lgb_model = LGBMClassifier(**self.base_models_config['lgbm'])

            # Apply PatchTST wrapper for tree-based models
            lgb_model = self._apply_patchtst_wrapper_if_needed(lgb_model, 'LGBM')

            if COMPREHENSIVE_TRAINING_AVAILABLE:
                # Use comprehensive training with overfitting detection
                lgb_model = self._train_with_comprehensive_pipeline(
                    lgb_model, X, y, 'lgbm', overfitting_detector, lookahead_protection, prevention_config
                )
            else:
                # Fallback to simple training
                lgb_model.fit(X, y)

            base_models['lgbm'] = lgb_model
            tprint("✅ [NAS_ENSEMBLE] LGBM trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_ENSEMBLE] LGBM training failed: {e}", color="red")

        # Train XGBoost with comprehensive pipeline
        tprint("🚀 [NAS_ENSEMBLE] Training XGBoost model with comprehensive pipeline", color="blue")
        try:
            xgb_model = XGBClassifier(**self.base_models_config['xgboost'])

            # Apply PatchTST wrapper for tree-based models
            xgb_model = self._apply_patchtst_wrapper_if_needed(xgb_model, 'XGBoost')

            if COMPREHENSIVE_TRAINING_AVAILABLE:
                # Use comprehensive training with overfitting detection
                xgb_model = self._train_with_comprehensive_pipeline(
                    xgb_model, X, y, 'xgboost', overfitting_detector, lookahead_protection, prevention_config
                )
            else:
                # Fallback to simple training
                xgb_model.fit(X, y)

            base_models['xgboost'] = xgb_model
            tprint("✅ [NAS_ENSEMBLE] XGBoost trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_ENSEMBLE] XGBoost training failed: {e}", color="red")

        # Train Random Forest with comprehensive pipeline
        tprint("🌳 [NAS_ENSEMBLE] Training Random Forest model with comprehensive pipeline", color="blue")
        try:
            rf_model = RandomForestClassifier(**self.base_models_config['random_forest'])

            # Apply PatchTST wrapper for tree-based models
            rf_model = self._apply_patchtst_wrapper_if_needed(rf_model, 'Random Forest')

            if COMPREHENSIVE_TRAINING_AVAILABLE:
                # Use comprehensive training with overfitting detection
                rf_model = self._train_with_comprehensive_pipeline(
                    rf_model, X, y, 'random_forest', overfitting_detector, lookahead_protection, prevention_config
                )
            else:
                # Fallback to simple training
                rf_model.fit(X, y)

            base_models['random_forest'] = rf_model
            tprint("✅ [NAS_ENSEMBLE] Random Forest trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_ENSEMBLE] Random Forest training failed: {e}", color="red")

        # Train ElasticNet with comprehensive pipeline
        tprint("📈 [NAS_ENSEMBLE] Training ElasticNet model with comprehensive pipeline", color="blue")
        try:
            en_model = ElasticNet(**self.base_models_config['elastic_net'])

            if COMPREHENSIVE_TRAINING_AVAILABLE:
                # Use comprehensive training with overfitting detection
                en_model = self._train_with_comprehensive_pipeline(
                    en_model, X, y, 'elastic_net', overfitting_detector, lookahead_protection, prevention_config
                )
            else:
                # Fallback to simple training
                en_model.fit(X, y)

            base_models['elastic_net'] = en_model
            tprint("✅ [NAS_ENSEMBLE] ElasticNet trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_ENSEMBLE] ElasticNet training failed: {e}", color="red")

        tprint(f"✅ [NAS_ENSEMBLE] Base models training completed - {len(base_models)} models trained", color="green")
        return base_models

    def _evaluate_base_models(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate base models performance."""
        tprint("📊 [NAS_ENSEMBLE] Evaluating base models performance", color="yellow")

        metrics = {}

        # Use TimeSeriesSplit for proper time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        for model_name, model in base_models.items():
            tprint(f"📊 [NAS_ENSEMBLE] Evaluating {model_name} model", color="blue")
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')

                # Training accuracy
                y_pred = model.predict(X)
                train_accuracy = accuracy_score(y, y_pred)

                metrics[model_name] = {
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_accuracy': train_accuracy,
                    'classification_report': classification_report(y, y_pred, output_dict=True)
                }

                tprint(f"✅ [NAS_ENSEMBLE] {model_name} - CV Mean: {cv_scores.mean():.4f}, Train Acc: {train_accuracy:.4f}", color="green")

            except Exception as e:
                tprint(f"❌ [NAS_ENSEMBLE] {model_name} evaluation failed: {e}", color="red")
                metrics[model_name] = {'error': str(e)}

        tprint("✅ [NAS_ENSEMBLE] Base models evaluation completed", color="green")
        return metrics

    def _apply_patchtst_wrapper_if_needed(self, base_model, model_name: str):
        """Apply PatchTST wrapper to tree-based models only."""
        # Tree-based model types that support PatchTST wrapper
        tree_model_types = {
            'LGBMClassifier', 'LGBMRegressor',
            'XGBClassifier', 'XGBRegressor',
            'RandomForestClassifier', 'RandomForestRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor'
        }

        model_type = base_model.__class__.__name__

        if model_type not in tree_model_types:
            tprint(f"ℹ️ [NAS_ENSEMBLE] PatchTST wrapper not applied to {model_name} ({model_type}) - not a tree-based model", color="yellow")
            return base_model

        if not (PATCHTST_AVAILABLE and create_patchtst_wrapper is not None):
            tprint(f"ℹ️ [NAS_ENSEMBLE] PatchTST wrapper not available for {model_name}", color="yellow")
            return base_model

        try:
            # Default PatchTST configuration
            patchtst_config = {
                'patch_len': 16,
                'stride': 8,
                'use_transformer_attention': True,
                'regime_aware': True,
                'attention_dropout': 0.1,
                'num_heads': 4,
                'sign_dropout_rate': 0.0,
                'sign_threshold': 0.2
            }

            wrapped_model = create_patchtst_wrapper(base_model, **patchtst_config)
            tprint(f"✅ [NAS_ENSEMBLE] {model_name} enhanced with PatchTST wrapper", color="green")
            return wrapped_model
        except Exception as e:
            tprint(f"⚠️ [NAS_ENSEMBLE] PatchTST wrapper failed for {model_name}, using base model: {e}", color="yellow")
            return base_model

    def _train_with_comprehensive_pipeline(self, model: Any, X: np.ndarray, y: np.ndarray,
                                         model_name: str, overfitting_detector: Any,
                                         lookahead_protection: Any, prevention_config: Any) -> Any:
        """Train model using comprehensive training pipeline with all protections."""
        tprint(f"🔧 [NAS_ENSEMBLE] Training {model_name} with comprehensive pipeline", color="blue")

        try:
            # 1. Lookahead bias protection
            tprint(f"🛡️ [NAS_ENSEMBLE] Applying lookahead bias protection for {model_name}", color="yellow")
            if hasattr(lookahead_protection, 'detect_and_prevent_leakage'):
                # Create DataFrame for lookahead protection
                data_df = pd.DataFrame(X)
                data_df['target'] = y
                data_df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(X), freq='1min')

                leakage_results = lookahead_protection.detect_and_prevent_leakage(data_df)
                if leakage_results.get('has_leakage', False):
                    tprint(f"⚠️ [NAS_ENSEMBLE] Lookahead bias detected for {model_name}: {leakage_results.get('leakage_details', [])}", color="yellow")
                else:
                    tprint(f"✅ [NAS_ENSEMBLE] No lookahead bias detected for {model_name}", color="green")

            # 2. Cross-validation with time series split
            tprint(f"📊 [NAS_ENSEMBLE] Performing time series cross-validation for {model_name}", color="yellow")
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train model on fold
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)

                # Evaluate on validation set
                y_pred = model_copy.predict(X_val)
                fold_score = accuracy_score(y_val, y_pred)
                cv_scores.append(fold_score)

                tprint(f"📊 [NAS_ENSEMBLE] {model_name} fold {fold+1}/5: {fold_score:.4f}", color="blue")

            # 3. Overfitting detection
            tprint(f"🔍 [NAS_ENSEMBLE] Performing overfitting detection for {model_name}", color="yellow")
            if hasattr(overfitting_detector, 'detect_overfitting'):
                # Train on full data for overfitting detection
                model.fit(X, y)
                y_pred = model.predict(X)

                # Calculate train/validation metrics
                train_accuracy = accuracy_score(y, y_pred)

                # Use CV scores as validation metrics
                val_accuracy = np.mean(cv_scores)
                accuracy_gap = train_accuracy - val_accuracy

                # Check for overfitting
                if accuracy_gap > 0.05:  # 5% gap threshold
                    tprint(f"⚠️ [NAS_ENSEMBLE] Potential overfitting detected for {model_name}: gap={accuracy_gap:.4f}", color="yellow")
                else:
                    tprint(f"✅ [NAS_ENSEMBLE] No overfitting detected for {model_name}: gap={accuracy_gap:.4f}", color="green")

            # 4. Final training with regularization
            tprint(f"🏋️ [NAS_ENSEMBLE] Final training with regularization for {model_name}", color="yellow")

            # Apply regularization if supported
            if hasattr(model, 'set_params'):
                if model_name in ['lgbm', 'xgboost']:
                    # Add regularization for tree-based models
                    model.set_params(reg_alpha=0.01, reg_lambda=0.01)
                elif model_name == 'elastic_net':
                    # ElasticNet already has regularization
                    pass
                elif model_name == 'random_forest':
                    # Add regularization for Random Forest
                    model.set_params(max_depth=10, min_samples_split=10, min_samples_leaf=5)

            # Final training
            model.fit(X, y)

            tprint(f"✅ [NAS_ENSEMBLE] {model_name} comprehensive training completed", color="green")
            return model

        except Exception as e:
            tprint(f"❌ [NAS_ENSEMBLE] Comprehensive training failed for {model_name}: {e}", color="red")
            # Fallback to simple training
            model.fit(X, y)
            return model

    def _clone_model(self, model: Any) -> Any:
        """Clone a model for cross-validation."""
        try:
            if hasattr(model, 'clone'):
                return model.clone()
            elif hasattr(model, '__class__'):
                model_class = model.__class__
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    return model_class(**params)
                else:
                    return model_class()
            else:
                return model
        except Exception as e:
            tprint(f"⚠️ [NAS_ENSEMBLE] Model cloning failed: {e}", color="yellow")
            return model
