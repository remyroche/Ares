"""
Ensemble Training Step

Base class for ensemble training steps with common functionality.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time

from src.utils.ml_common.training.base_training_step import BaseTrainingStep
from src.utils.ml_common.config.base_training_config import EnsembleTrainingConfig
from src.utils.ml_common.ensembles import StackingEnsembleManager, StackingEnsembleConfig

logger = logging.getLogger(__name__)


class EnsembleTrainingStep(BaseTrainingStep):
    """
    Base class for ensemble training steps.
    
    This class provides common functionality for training ensemble models,
    including base model creation, ensemble training, and evaluation.
    """
    
    def __init__(self, config: EnsembleTrainingConfig):
        """
        Initialize ensemble training step.
        
        Args:
            config: Ensemble training configuration
        """
        super().__init__(config)
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Ensemble specific results
        self.ensemble_models = {}
        self.ensemble_metadata = {}
        
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
        Train ensemble models for each regime.
        
        Args:
            regime_data: Prepared data for each regime
            base_models: Pre-trained base models for each regime
            feature_names: Names of input features
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing training results for each regime
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
                'augmented': data['augmented'],
                'hmm_states': data.get('hmm_states'),
                'base_models': list(regime_base_models.keys()),
                'training_time': time.time()
            }
            
            self.logger.info(f"✅ Regime {regime} ensemble trained")
        
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
        Execute ensemble training step.
        
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
            
        except Exception as e:
            return self._handle_training_error(e, "ensemble training")
    
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