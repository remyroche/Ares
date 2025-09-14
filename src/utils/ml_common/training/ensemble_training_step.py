"""
Ensemble Training Step

Base class for ensemble training steps with common functionality.
Enhanced with vectorized training capabilities for improved performance.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time

from src.utils.ml_common.training.base_training_step import BaseTrainingStep
from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
from src.utils.ml_common.ensembles import StackingEnsembleManager, StackingEnsembleConfig

# Import vectorized training manager
try:
    from src.utils.ml_common.training.vectorized_training_manager import VectorizedTrainingManager
    VECTORIZED_TRAINING_AVAILABLE = True
except ImportError:
    VECTORIZED_TRAINING_AVAILABLE = False
    VectorizedTrainingManager = None

logger = logging.getLogger(__name__)


class EnsembleTrainingStep(BaseTrainingStep):
    """
    Base class for ensemble training steps.
    
    This class provides common functionality for training ensemble models,
    including base model creation, ensemble training, and evaluation.
    """
    
    def __init__(self, config: EnsembleTrainingConfig, enable_vectorization: bool = True):
        """
        Initialize ensemble training step with optional vectorization.

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

        self.logger.info("✅ Ensemble Training Step initialized")
    
    def create_ensemble_models(
        self,
        base_models: Dict[str, Any],
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Create ensemble models with specified meta-learner.
        
        Args:
            base_models: Dictionary of base models
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing ensemble models
        """
        ensembles = {}
        
        if is_classification:
            # Stacking ensemble with specified meta-learner
            meta_learner = self.training_utils.create_model(
                model_type=self.config.meta_model,
                model_name=f"{self.config.model_name}_meta_learner",
                model_params=self._get_meta_model_params()
            )
            
            from sklearn.ensemble import StackingClassifier
            ensembles['stacking_ensemble'] = StackingClassifier(
                estimators=list(base_models.items()),
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                n_jobs=-1
            )
            
        else:
            # Stacking ensemble with specified meta-learner
            meta_learner = self.training_utils.create_model(
                model_type=self.config.meta_model,
                model_name=f"{self.config.model_name}_meta_learner",
                model_params=self._get_meta_model_params()
            )
            
            from sklearn.ensemble import StackingRegressor
            ensembles['stacking_ensemble'] = StackingRegressor(
                estimators=list(base_models.items()),
                final_estimator=meta_learner,
                cv=self.config.cv_folds,
                n_jobs=-1
            )
        
        return ensembles
    
    def train_ensemble_models(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble models.
        
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
            
            # Train ensemble
            start_time = time.time()
            ensemble.fit(X, y)
            training_time = time.time() - start_time
            
            ensemble_results[name] = {
                'ensemble': ensemble,
                'base_models': base_models,
                'training_time': training_time,
                'config': {
                    'meta_model': self.config.meta_model,
                    'cv_folds': self.config.cv_folds,
                    'base_models': list(base_models.keys())
                }
            }
            
            self.logger.info(f"✅ {name} completed in {training_time:.2f}s")
        
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
        Optimize ensemble using HPO.
        
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
        
        # Create ensemble configuration
        ensemble_config = StackingEnsembleConfig(
            base_models=base_models,
            meta_model_type=self.config.meta_model,
            meta_model_params=self._get_meta_model_params(),
            enable_cross_validation=self.config.enable_cross_validation,
            cv_folds=self.config.cv_folds,
            validation_split=self.config.validation_split
        )
        
        # Create ensemble manager
        ensemble_manager = StackingEnsembleManager(ensemble_config)
        
        # Apply overfitting prevention
        if self.config.enable_overfitting_prevention:
            for model_name, model in base_models.items():
                base_models[model_name] = self.training_utils.overfitting_prevention.apply_regularization(
                    model, model_name
                )
        
        # Train ensemble
        start_time = time.time()
        ensemble_manager.train(X, y)
        training_time = time.time() - start_time
        
        return {
            'ensemble_manager': ensemble_manager,
            'base_models': base_models,
            'regime': regime,
            'training_time': training_time,
            'config': ensemble_config
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
        Train single ensemble without HPO.
        
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
        
        # Create ensemble configuration
        ensemble_config = StackingEnsembleConfig(
            base_models=base_models,
            meta_model_type=self.config.meta_model,
            meta_model_params=self._get_meta_model_params(),
            enable_cross_validation=self.config.enable_cross_validation,
            cv_folds=self.config.cv_folds,
            validation_split=self.config.validation_split
        )
        
        # Create ensemble manager
        ensemble_manager = StackingEnsembleManager(ensemble_config)
        
        # Apply overfitting prevention
        if self.config.enable_overfitting_prevention:
            for model_name, model in base_models.items():
                base_models[model_name] = self.training_utils.overfitting_prevention.apply_regularization(
                    model, model_name
                )
        
        # Train ensemble
        start_time = time.time()
        ensemble_manager.train(X, y)
        training_time = time.time() - start_time
        
        return {
            'ensemble_manager': ensemble_manager,
            'base_models': base_models,
            'regime': regime,
            'training_time': training_time,
            'config': ensemble_config
        }
    
    def _get_meta_model_params(self) -> Dict[str, Any]:
        """
        Get default parameters for meta model.
        
        Returns:
            Dictionary of meta model parameters
        """
        return {
            'alpha': 1.0,
            'solver': 'auto',
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
            self.logger.warning("⚠️ No base models provided, using mock models")
            base_models = self._create_mock_base_models()

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
                model_type=self.config.model_name,
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
                model_type=self.config.model_name,
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
        Evaluate ensemble performance per regime.
        
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
            
            # Make predictions
            y_pred = ensemble_manager.predict(regime_X)
            
            # Calculate metrics
            metrics = self.evaluation_utils.calculate_metrics(
                y_true=regime_y,
                y_pred=y_pred,
                metrics=self.config.evaluation_metrics,
                is_classification=is_classification
            )
            
            evaluation_results[regime] = metrics
        
        return evaluation_results