"""
Ensemble Training Step - Enhanced with Overfitting Prevention and Lookahead Bias Detection

Base class for ensemble training steps with comprehensive ML utilities integration.
Enhanced Features:
- Purged cross-validation for temporal data integrity
- Early stopping for all supported models
- Lookahead bias detection and prevention
- Enhanced regularization parameters
- Overfitting monitoring and detection
- Walk-forward validation
- Ensemble diversity metrics
- Vectorized training capabilities for improved performance
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import logging
import time
from datetime import datetime, timedelta
import warnings

from .base_training_step import BaseTrainingStep
from ..config.base_training_config import EnsembleTrainingConfig
from ..ensembles import StackingEnsembleManager, StackingEnsembleConfig

# Import vectorized training manager
try:
    from .vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False
    VectorizedTrainingManager = None

# Enhanced training utilities are now available from BaseTrainingStep
# No need to import separately - use self.enhanced_training_available

logger = logging.getLogger(__name__)


class EnsembleTrainingStep(BaseTrainingStep):
    """
    Enhanced base class for ensemble training steps with comprehensive ML utilities.
    
    This class provides common functionality for training ensemble models,
    including base model creation, ensemble training, and evaluation.
    
    Enhanced Features:
    - Overfitting prevention and detection
    - Lookahead bias detection and prevention
    - Enhanced regularization parameters
    - Early stopping for all supported models
    - Purged cross-validation for temporal data
    - Walk-forward validation
    - Ensemble diversity monitoring
    - Vectorized training capabilities
    """
    
    def __init__(self, config: EnsembleTrainingConfig, enable_vectorization: bool = True):
        """
        Initialize enhanced ensemble training step with optional vectorization.

        Args:
            config: Ensemble training configuration
            enable_vectorization: Whether to use vectorized training when available
        """
        super().__init__(config)
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)

        # Ensemble specific results
        self.ensemble_models = {}
        self.ensemble_metadata = {}
        
        # Initialize enhanced training utilities
        if self.enhanced_training_available:
            self._initialize_enhanced_training_utilities()

        # Vectorized training manager
        self.enable_vectorization = enable_vectorization and VECTORIZED_TRAINING_AVAILABLE
        if self.enable_vectorization:
            try:
                self.vectorized_manager = VectorizedTrainingManager()
                self.logger.info("🚀 Vectorized training enabled and initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Vectorized training enabled but failed to initialize: {e}")
                self.vectorized_manager = None
                self.enable_vectorization = False
        else:
            self.vectorized_manager = None
            if VECTORIZED_TRAINING_AVAILABLE:
                self.logger.info("ℹ️ Vectorized training available but disabled (enable_vectorization=False)")
            else:
                self.logger.info("⚠️ Vectorized training not available (import failed)")

        self.logger.info("✅ Enhanced Ensemble Training Step initialized")
    
    def _initialize_enhanced_training_utilities(self):
        """Initialize enhanced training utilities for ensemble training (inherited from BaseTrainingStep)."""
        # Enhanced training utilities are already available from base class
        # Can access via self.enhanced_training_utils, self.training_enhancer, etc.
        self.logger.info("✅ Enhanced training utilities initialized for ensemble training")
        try:
            # Create enhanced training configuration for Ensemble
            self.enhanced_training_config = TrainingIntegrationConfig(
                enable_early_stopping=True,
                enable_purged_cv=True,
                enable_lookahead_detection=True,
                enable_temporal_splits=True,
                enable_regularization=True,
                enable_overfitting_monitoring=True,
                enable_ensemble_diversity=True,  # Enable for ensemble
                model_type='auto'
            )
            
            # Initialize training enhancer
            self.training_enhancer = TrainingStepEnhancer(self.enhanced_training_config)
            
            self.logger.info("✅ Enhanced training utilities initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Enhanced training utilities initialization failed: {e}")
            self.enhanced_training_config = None
            self.training_enhancer = None
    
    def create_ensemble_models(
        self,
        base_models: Dict[str, Any],
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Create ensemble models using OOF stacking with temporal validation.

        Args:
            base_models: Dictionary of base models
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing ensemble models
        """
        ensembles = {}

        # Use OOF stacking ensemble for proper temporal validation
        ensemble_name = f"{self.config.model_name}_oof_stacking_ensemble"

        # Create OOF stacking ensemble
        oof_ensemble = self.training_utils.create_oof_stacking_ensemble(
            base_models=base_models,
            ensemble_name=ensemble_name,
            n_outputs=1 if not isinstance(list(base_models.values())[0], dict) else len(base_models),
            enable_temporal_validation=True,
            cv_folds=self.config.cv_folds
        )

        ensembles['oof_stacking_ensemble'] = oof_ensemble

        return ensembles
    
    def train_ensemble_models(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble models using OOF stacking with comprehensive validation.

        Args:
            base_models: Dictionary of base models
            X: Input features
            y: Target values
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing training results
        """
        # Create ensemble models
        ensemble_models = self.create_ensemble_models(base_models, is_classification)

        # Train ensemble models
        ensemble_results = {}

        for name, ensemble in ensemble_models.items():
            self.logger.info(f"🔄 Training ensemble model: {name}")

            # Use comprehensive training with validation
            start_time = time.time()
            trained_ensemble, validation_results = self.training_utils.train_oof_stacking_ensemble(
                ensemble_manager=ensemble,
                X=X,
                y=y,
                model_name=name,
                model_type="stacking",
                timestamps=None,  # Add timestamp support if available
                feature_names=None  # Add feature names if available
            )
            training_time = time.time() - start_time

            ensemble_results[name] = {
                'ensemble': trained_ensemble,
                'base_models': base_models,
                'training_time': training_time,
                'validation_results': validation_results,
                'config': {
                    'meta_model': self.config.meta_model,
                    'cv_folds': self.config.cv_folds,
                    'base_models': list(base_models.keys()),
                    'oof_validation': True,
                    'temporal_validation': True
                }
            }

            self.logger.info(f"✅ {name} completed in {training_time:.2f}s with OOF validation")

            # Log validation results with comprehensive details
            if validation_results['valid']:
                self.logger.info(f"✅ Ensemble validation passed: {validation_results['validation_score']:.4f}")
                # Log additional validation details if available
                if 'enhanced_validation' in validation_results:
                    enhanced = validation_results['enhanced_validation']
                    self.logger.info(f"📊 Enhanced validation score: {enhanced.get('validation_score', 'N/A')}")
                    self.logger.info(f"📈 Performance stability: {enhanced.get('performance_stability', 'N/A')}")
                    self.logger.info(f"🔍 Validation reliability: {enhanced.get('validation_reliability', 'N/A')}")
            else:
                self.logger.warning(f"⚠️ Ensemble validation failed: {len(validation_results.get('critical_issues', []))} issues")
                for issue in validation_results.get('critical_issues', []):
                    self.logger.error(f"❌ Critical issue: {issue}")
                for warning in validation_results.get('warnings', []):
                    self.logger.warning(f"⚠️ Warning: {warning}")

        return ensemble_results
    
    def train_regime_ensembles(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        base_models: Optional[Dict[int, Dict[str, Any]]] = None,
        feature_names: Optional[List[str]] = None,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble models for each regime with optional vectorized parallel processing.

        Args:
            regime_data: Prepared data for each regime
            base_models: Pre-trained base models for each regime
            feature_names: Names of input features
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing training results for each regime
        """
        # Use vectorized parallel training if available
        if self.enable_vectorization and self.vectorized_manager and len(regime_data) > 1:
            self.logger.info("🚀 VECTORIZED: Using parallel regime training")
            return self._train_regime_ensembles_parallel(
                regime_data, base_models, feature_names, is_classification
            )

        # Fall back to sequential training
        self.logger.info("🔄 Using sequential regime training")
        return self._train_regime_ensembles_sequential(
            regime_data, base_models, feature_names, is_classification
        )

    def _train_regime_ensembles_parallel(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        base_models: Optional[Dict[int, Dict[str, Any]]] = None,
        feature_names: Optional[List[str]] = None,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        VECTORIZED: Train ensemble models for each regime in parallel.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        regime_ensembles = {}
        regime_metadata = {}

        # Filter out regimes with insufficient data
        valid_regimes = {
            regime: data for regime, data in regime_data.items()
            if not data.get('use_global', False)
        }

        if not valid_regimes:
            self.logger.warning("⚠️ No valid regimes found for training")
            return {'ensembles': {}, 'metadata': {}}

        self.logger.info(f"🔄 Training ensembles for {len(valid_regimes)} regimes in parallel...")

        def train_single_regime(regime: int, data: Dict[str, np.ndarray]) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
            """Train ensemble for a single regime."""
            regime_start_time = time.time()

            # Get base models for this regime
            if base_models and regime in base_models:
                regime_base_models = base_models[regime]
            else:
                regime_base_models = self._create_base_models_for_regime(regime, data['X'], data['y'])

            # Train ensemble for this regime
            try:
                if self.config.enable_hpo:
                    ensemble_result = self._optimize_ensemble(
                        regime_base_models, data['X'], data['y'], regime, feature_names, is_classification
                    )
                else:
                    ensemble_result = self._train_single_ensemble(
                        regime_base_models, data['X'], data['y'], regime, feature_names, is_classification
                    )

                # Store regime metadata
                regime_metadata_result = {
                    'samples': data['samples'],
                    'augmented': data.get('augmented', False),
                    'hmm_states': data.get('hmm_states'),
                    'base_models': list(regime_base_models.keys()),
                    'training_time': time.time() - regime_start_time,
                    'success': True
                }

                return regime, ensemble_result, regime_metadata_result

            except Exception as e:
                self.logger.warning(f"⚠️ Regime {regime} training failed: {e}")
                regime_metadata_result = {
                    'samples': data['samples'],
                    'error': str(e),
                    'training_time': time.time() - regime_start_time,
                    'success': False
                }
                return regime, {'error': str(e)}, regime_metadata_result

        # Train regimes in parallel
        with ThreadPoolExecutor(max_workers=min(self.vectorized_manager.max_workers, len(valid_regimes))) as executor:
            futures = [
                executor.submit(train_single_regime, regime, data)
                for regime, data in valid_regimes.items()
            ]

            for future in as_completed(futures):
                regime, ensemble_result, metadata = future.result()
                regime_ensembles[regime] = ensemble_result
                regime_metadata[regime] = metadata

                if metadata.get('success', False):
                    self.logger.info(f"✅ Regime {regime} ensemble trained in {metadata['training_time']:.2f}s")
                else:
                    self.logger.warning(f"❌ Regime {regime} training failed")

        # Log skipped regimes
        skipped_regimes = len(regime_data) - len(valid_regimes)
        if skipped_regimes > 0:
            self.logger.info(f"⏭️ Skipped {skipped_regimes} regimes (insufficient data)")

        return {
            'ensembles': regime_ensembles,
            'metadata': regime_metadata
        }

    def _train_regime_ensembles_sequential(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        base_models: Optional[Dict[int, Dict[str, Any]]] = None,
        feature_names: Optional[List[str]] = None,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble models for each regime sequentially (fallback method).
        """
        regime_ensembles = {}
        regime_metadata = {}

        for regime, data in regime_data.items():
            if data.get('use_global', False):
                self.logger.info(f"⏭️ Skipping regime {regime} (insufficient data, will use global model)")
                continue

            self.logger.info(f"🔄 Training ensemble for regime {regime} ({data['samples']} samples)...")

            # Get base models for this regime
            if base_models and regime in base_models:
                regime_base_models = base_models[regime]
            else:
                self.logger.warning(f"⚠️ No base models found for regime {regime}, creating new ones")
                regime_base_models = self._create_base_models_for_regime(regime, data['X'], data['y'])

            # Train ensemble for this regime
            try:
                if self.config.enable_hpo:
                    ensemble_result = self._optimize_ensemble(
                        regime_base_models, data['X'], data['y'], regime, feature_names, is_classification
                    )
                else:
                    ensemble_result = self._train_single_ensemble(
                        regime_base_models, data['X'], data['y'], regime, feature_names, is_classification
                    )

                regime_ensembles[regime] = ensemble_result

                # Store regime metadata
                regime_metadata[regime] = {
                    'samples': data['samples'],
                    'augmented': data.get('augmented', False),
                    'hmm_states': data.get('hmm_states'),
                    'base_models': list(regime_base_models.keys()),
                    'training_time': time.time(),
                    'success': True
                }

                self.logger.info(f"✅ Regime {regime} ensemble trained")

            except Exception as e:
                self.logger.warning(f"⚠️ Regime {regime} training failed: {e}")
                regime_ensembles[regime] = {'error': str(e)}
                regime_metadata[regime] = {
                    'samples': data['samples'],
                    'error': str(e),
                    'success': False
                }

        return {
            'ensembles': regime_ensembles,
            'metadata': regime_metadata
        }
    
    def _create_base_models_for_regime(
        self,
        regime: int,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create base models for regime if not provided.
        
        Args:
            regime: Regime identifier
            X: Input features
            y: Target values
            
        Returns:
            Dictionary of base models
        """
        base_models = {}
        
        for model_type in self.config.base_models:
            model = self.training_utils.create_model(
                model_type=model_type,
                model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}",
                model_params=self.training_utils.get_model_params(model_type)
            )
            
            model.fit(X, y)
            base_models[model_type] = model
        
        return base_models
    
    def _optimize_ensemble(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime: int,
        feature_names: Optional[List[str]],
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Optimize ensemble using OOF stacking with HPO and comprehensive validation.

        Args:
            base_models: Base models for ensemble
            X: Input features
            y: Target values
            regime: Regime identifier
            feature_names: Names of input features
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing optimization results
        """
        self.logger.debug(f"🔄 Optimizing ensemble for regime {regime}...")

        # Create OOF stacking ensemble
        ensemble_name = f"hmm_ensemble_regime_{regime}_oof_stacking"
        oof_ensemble = self.training_utils.create_oof_stacking_ensemble(
            base_models=base_models,
            ensemble_name=ensemble_name,
            n_outputs=1 if not isinstance(list(base_models.values())[0], dict) else len(base_models),
            enable_temporal_validation=True,
            cv_folds=self.config.cv_folds
        )

        # Apply overfitting prevention
        if self.config.enable_overfitting_prevention:
            for output_name, models in base_models.items():
                for model_name, model in models.items():
                    base_models[output_name][model_name] = self.training_utils.overfitting_prevention.apply_regularization(
                        model, model_name
                    )

        # Train ensemble with comprehensive validation
        start_time = time.time()
        trained_ensemble, validation_results = self.training_utils.train_oof_stacking_ensemble(
            ensemble_manager=oof_ensemble,
            X=X,
            y=y,
            model_name=f"regime_{regime}_ensemble",
            model_type="stacking",
            timestamps=None,
            feature_names=feature_names
        )
        training_time = time.time() - start_time

        return {
            'ensemble_manager': trained_ensemble,
            'base_models': base_models,
            'regime': regime,
            'training_time': training_time,
            'validation_results': validation_results,
            'config': {
                'ensemble_name': ensemble_name,
                'oof_validation': True,
                'temporal_validation': True,
                'cv_folds': self.config.cv_folds
            }
        }
    
    def _train_single_ensemble(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime: int,
        feature_names: Optional[List[str]],
        is_classification: bool
    ) -> Dict[str, Any]:
        """
        Train single ensemble without HPO using OOF stacking.

        Args:
            base_models: Base models for ensemble
            X: Input features
            y: Target values
            regime: Regime identifier
            feature_names: Names of input features
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing training results
        """
        self.logger.debug(f"🔄 Training ensemble for regime {regime} (no HPO)...")

        # Create OOF stacking ensemble
        ensemble_name = f"hmm_ensemble_regime_{regime}_oof_stacking"
        oof_ensemble = self.training_utils.create_oof_stacking_ensemble(
            base_models=base_models,
            ensemble_name=ensemble_name,
            n_outputs=1 if not isinstance(list(base_models.values())[0], dict) else len(base_models),
            enable_temporal_validation=True,
            cv_folds=self.config.cv_folds
        )

        # Apply overfitting prevention
        if self.config.enable_overfitting_prevention:
            for output_name, models in base_models.items():
                for model_name, model in models.items():
                    base_models[output_name][model_name] = self.training_utils.overfitting_prevention.apply_regularization(
                        model, model_name
                    )

        # Train ensemble with comprehensive validation
        start_time = time.time()
        trained_ensemble, validation_results = self.training_utils.train_oof_stacking_ensemble(
            ensemble_manager=oof_ensemble,
            X=X,
            y=y,
            model_name=f"regime_{regime}_ensemble",
            model_type="stacking",
            timestamps=None,
            feature_names=feature_names
        )
        training_time = time.time() - start_time

        return {
            'ensemble_manager': trained_ensemble,
            'base_models': base_models,
            'regime': regime,
            'training_time': training_time,
            'validation_results': validation_results,
            'config': {
                'ensemble_name': ensemble_name,
                'oof_validation': True,
                'temporal_validation': True,
                'cv_folds': self.config.cv_folds
            }
        }
    
    def _get_meta_model_params(self) -> Dict[str, Any]:
        """
        Get default parameters for meta model based on model type.
        
        Returns:
            Dictionary of meta model parameters
        """
        meta_model = self.config.meta_model
        
        if meta_model == 'ElasticNet':
            return {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 2000,  # Increased to prevent convergence warnings
                'random_state': 42,
                'tol': 1e-4  # Tolerance for convergence
            }
        elif meta_model == 'XGBoostClassifier':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        elif meta_model == 'CatBoostClassifier':
            return {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': False
            }
        else:
            # Fallback for unknown models
            return {
                'random_state': 42
            }
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute ensemble training step with optional vectorization.

        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            **kwargs: Additional arguments

        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting ensemble training step")
        start_time = time.time()

        try:
            # Use vectorized training if available and enabled
            if self.enable_vectorization and self.vectorized_manager:
                self.logger.info("🚀 VECTORIZED: Using vectorized ensemble training")
                return self._execute_vectorized_training(
                    X, y, regime_labels, feature_names, hmm_states, **kwargs
                )

            # Fallback to standard training
            self.logger.info("🔄 Using standard ensemble training")
            return self._execute_standard_training(
                X, y, regime_labels, feature_names, hmm_states, **kwargs
            )

        except Exception as e:
            return self._handle_training_error(e, "ensemble training")

    def _execute_vectorized_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        VECTORIZED: Execute ensemble training using vectorized manager.
        """
        self.logger.info("🚀 VECTORIZED: Starting vectorized ensemble training")

        # Get base models
        base_models = kwargs.get('base_models')
        if base_models is None or not base_models:
            self.logger.warning("⚠️ No base models provided; proceeding to standard training path")
            return self._execute_standard_training(
                X, y, regime_labels, feature_names, hmm_states, **kwargs
            )

        # Use vectorized ensemble training
        vectorized_results = self.vectorized_manager.vectorized_ensemble_training(
            X=X,
            y=y,
            regime_labels=regime_labels,
            base_models=base_models,
            model_types=self.config.model_types,
            is_classification=kwargs.get('is_classification', True),
            enable_hpo=self.config.enable_hpo,
            cv_folds=self.config.cv_folds
        )

        if 'error' in vectorized_results:
            self.logger.warning(f"⚠️ Vectorized training failed: {vectorized_results['error']}")
            return self._execute_standard_training(
                X, y, regime_labels, feature_names, hmm_states, **kwargs
            )

        # Extract results from vectorized training
        ensemble_results = vectorized_results['ensemble_results']
        evaluation_results = vectorized_results['evaluation_results']
        training_time = vectorized_results['training_time']
        vectorization_stats = vectorized_results['vectorization_stats']

        # Create final results in expected format
        results = self._create_final_results(
            models=ensemble_results,
            metadata={'vectorization_stats': vectorization_stats},
            evaluation_results=evaluation_results,
            training_time=training_time,
            additional_results={'vectorized': True, 'stats': vectorization_stats}
        )

        self.training_results = results

        # Log vectorization benefits
        speedup = vectorization_stats.get('speedup_estimate', 1.0)
        self.logger.info(f"🚀 VECTORIZED: Achieved {speedup:.2f}x speedup")
        self.logger.info(f"📊 VECTORIZED: Processed {vectorization_stats.get('regimes_processed', 0)} regimes")

        return results

    def _execute_standard_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute standard (non-vectorized) ensemble training.
        """
        start_time = time.time()

        # Step 1: Analyze regimes and prepare data
        self.logger.info("🔄 Step 1: Analyzing regimes and preparing data...")
        regime_analysis = self.analyze_regimes(regime_labels)
        regime_data = self.prepare_regime_data(X, y, regime_labels, regime_analysis, hmm_states)

        # Step 2: Train ensemble models for each regime
        self.logger.info("🔄 Step 2: Training ensemble models for each regime...")
        base_models = kwargs.get('base_models')
        regime_results = self.train_regime_ensembles(
            regime_data, base_models, feature_names,
            is_classification=kwargs.get('is_classification', True)
        )

        # Step 3: Save ensemble models
        if self.config.save_models:
            self.logger.info("🔄 Step 3: Saving trained ensemble models...")
            symbol = kwargs.get('symbol')
            exchange = kwargs.get('exchange')
            timeframe = kwargs.get('timeframe', self.config.timeframe)
            self._save_ensemble_models(regime_results, symbol, exchange, timeframe)

        # Step 4: Evaluate ensemble performance
        if self.config.enable_evaluation:
            self.logger.info("🔄 Step 4: Evaluating ensemble performance...")
            evaluation_results = self._evaluate_ensembles(
                regime_results, X, y, regime_labels,
                is_classification=kwargs.get('is_classification', True)
            )
        else:
            evaluation_results = {}

        # Create final results
        total_time = time.time() - start_time
        results = self._create_final_results(
            models=regime_results['ensembles'],
            metadata=regime_results['metadata'],
            evaluation_results=evaluation_results,
            training_time=total_time,
            additional_results={'regime_analysis': regime_analysis}
        )

        self.training_results = results

        # Log summary
        n_ensembles = len(regime_results['ensembles'])
        self._log_training_summary(results, f"Ensemble {self.config.model_name}", n_ensembles)

        return results
    
    def _save_ensemble_models(
        self,
        regime_results: Dict[str, Any],
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[int, List[str]]:
        """
        Save trained ensemble models for each regime.
        
        Args:
            regime_results: Training results for each regime
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            
        Returns:
            Dictionary containing saved model paths for each regime
        """
        saved_paths = {}
        
        for regime, ensemble_result in regime_results['ensembles'].items():
            # Save ensemble manager
            ensemble_manager = ensemble_result['ensemble_manager']
            model_paths = self.save_models(
                models={'ensemble': ensemble_manager},
                model_type=str(self.config.model_name),
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )
            
            saved_paths[regime] = model_paths
            
            # Save ensemble metadata
            ensemble_metadata = regime_results['metadata'][regime]
            self.save_metadata(
                metadata=ensemble_metadata,
                model_type=str(self.config.model_name),
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )
        
        return saved_paths
    
    def _evaluate_ensembles(
        self,
        regime_results: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble performance per regime using OOF predictions.

        Args:
            regime_results: Training results for each regime
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing evaluation results per regime
        """
        evaluation_results = {}

        for regime, ensemble_result in regime_results['ensembles'].items():
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]

            ensemble_manager = ensemble_result['ensemble_manager']

            # Use OOF predictions for evaluation (proper out-of-sample)
            if hasattr(ensemble_manager, 'get_oof_predictions'):
                oof_predictions = ensemble_manager.get_oof_predictions()
                oof_scores = ensemble_manager.get_oof_scores()

                # Aggregate OOF predictions for this regime
                if regime in oof_predictions and isinstance(oof_predictions[regime], dict) and len(oof_predictions[regime]) > 0:
                    # Use OOF predictions instead of new predictions
                    try:
                        y_pred = np.array(list(oof_predictions[regime].values())).mean(axis=0)
                    except Exception:
                        # Fallback: concatenate and average last axis if values are arrays
                        vals = list(oof_predictions[regime].values())
                        y_stack = np.column_stack([np.asarray(v).ravel() for v in vals])
                        y_pred = y_stack.mean(axis=1)

                    # Calculate metrics using OOF predictions
                    metrics = self.evaluation_utils.calculate_metrics(
                        y_true=regime_y,
                        y_pred=y_pred,
                        metrics=self.config.evaluation_metrics,
                        is_classification=is_classification
                    )

                    # Add OOF-specific metrics
                    metrics['oof_validation'] = True
                    metrics['oof_score'] = oof_scores.get(regime, 0.0)
                    metrics['evaluation_method'] = 'out_of_fold'

                    evaluation_results[regime] = metrics

                    self.logger.info(f"✅ Regime {regime} evaluated using OOF predictions: {oof_scores.get(regime, 0.0):.4f}")
                else:
                    # Fallback to regular predictions if OOF not available
                    self.logger.warning(f"⚠️ OOF predictions not available for regime {regime}, using regular predictions")
                    y_pred = ensemble_manager.predict(regime_X)

                    metrics = self.evaluation_utils.calculate_metrics(
                        y_true=regime_y,
                        y_pred=y_pred,
                        metrics=self.config.evaluation_metrics,
                        is_classification=is_classification
                    )

                    metrics['oof_validation'] = False
                    metrics['evaluation_method'] = 'in_sample'
                    evaluation_results[regime] = metrics
            else:
                # Fallback for non-OOF ensembles
                y_pred = ensemble_manager.predict(regime_X)

                metrics = self.evaluation_utils.calculate_metrics(
                    y_true=regime_y,
                    y_pred=y_pred,
                    metrics=self.config.evaluation_metrics,
                    is_classification=is_classification
                )

                metrics['oof_validation'] = False
                metrics['evaluation_method'] = 'in_sample'
                evaluation_results[regime] = metrics

                self.logger.warning(f"⚠️ OOF evaluation not available for regime {regime}")

        return evaluation_results