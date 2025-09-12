"""
Per-Regime Training Step

Base class for per-regime training steps with common functionality.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time

from src.utils.ml_common.training.base_training_step import BaseTrainingStep
from src.utils.ml_common.config.base_training_config import PerRegimeTrainingConfig

logger = logging.getLogger(__name__)


class PerRegimeTrainingStep(BaseTrainingStep):
    """
    Base class for per-regime training steps.
    
    This class provides common functionality for training models on a per-regime basis,
    including regime analysis, data preparation, and per-regime model training.
    """
    
    def __init__(self, config: PerRegimeTrainingConfig):
        """
        Initialize per-regime training step.
        
        Args:
            config: Per-regime training configuration
        """
        super().__init__(config)
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Per-regime specific results
        self.regime_models = {}
        self.regime_metadata = {}
        
        self.logger.info("✅ Per-Regime Training Step initialized")
    
    def train_regime_models(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train models for each regime.
        
        Args:
            regime_data: Prepared data for each regime
            feature_names: Names of input features
            
        Returns:
            Dictionary containing training results for each regime
        """
        regime_models = {}
        regime_metadata = {}
        
        for regime, data in regime_data.items():
            if data.get('use_global', False):
                self.logger.info(f"⏭️ Skipping regime {regime} (insufficient data, will use global model)")
                continue
            
            self.logger.info(f"🔄 Training models for regime {regime} ({data['samples']} samples)...")
            
            # Train each model type for this regime
            regime_model_results = {}
            
            for model_type in self.config.model_types:
                self.logger.info(f"🔄 Training {model_type} for regime {regime}...")
                
                # Perform HPO if enabled
                if self.config.enable_hpo:
                    search_space = self.config.hpo_search_spaces.get(model_type, {})
                    optimized_model = self.training_utils.optimize_model_with_hpo(
                        model_type=model_type,
                        X=data['X'],
                        y=data['y'],
                        search_space=search_space,
                        model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
                    )
                else:
                    optimized_model = self.training_utils.train_single_model(
                        model_type=model_type,
                        X=data['X'],
                        y=data['y'],
                        model_name=f"{self.config.model_name}_{model_type.lower()}_regime_{regime}"
                    )
                
                regime_model_results[model_type] = optimized_model
            
            regime_models[regime] = regime_model_results
            
            # Store regime metadata
            regime_metadata[regime] = {
                'samples': data['samples'],
                'augmented': data['augmented'],
                'hmm_states': data.get('hmm_states'),
                'models_trained': list(regime_model_results.keys()),
                'training_time': time.time()
            }
            
            self.logger.info(f"✅ Regime {regime} models trained: {list(regime_model_results.keys())}")
        
        return {
            'models': regime_models,
            'metadata': regime_metadata
        }
    
    def evaluate_regime_models(
        self,
        regime_results: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance per regime.
        
        Args:
            regime_results: Training results for each regime
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing evaluation results per regime
        """
        return self.evaluation_utils.evaluate_regime_performance(
            models={regime: {model_type: result['model'] for model_type, result in models.items()}
                   for regime, models in regime_results['models'].items()},
            X=X,
            y=y,
            regime_labels=regime_labels,
            metrics=self.config.evaluation_metrics,
            is_classification=is_classification
        )
    
    def save_regime_models(
        self,
        regime_results: Dict[str, Any],
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[int, List[str]]:
        """
        Save trained models for each regime.
        
        Args:
            regime_results: Training results for each regime
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            
        Returns:
            Dictionary containing saved model paths for each regime
        """
        saved_paths = {}
        
        for regime, models in regime_results['models'].items():
            # Extract models from results
            model_dict = {model_type: result['model'] for model_type, result in models.items()}
            
            # Save models for this regime
            model_paths = self.save_models(
                models=model_dict,
                model_type=self.config.model_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )
            
            saved_paths[regime] = model_paths
            
            # Save regime metadata
            regime_metadata = regime_results['metadata'][regime]
            self.save_metadata(
                metadata=regime_metadata,
                model_type=self.config.model_name,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                regime=regime
            )
        
        return saved_paths
    
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
        Execute per-regime training step.
        
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
        self.logger.info("🚀 Starting per-regime training step")
        start_time = time.time()
        
        try:
            # Step 1: Analyze regimes and prepare data
            self.logger.info("🔄 Step 1: Analyzing regimes and preparing data...")
            regime_analysis = self.analyze_regimes(regime_labels)
            regime_data = self.prepare_regime_data(X, y, regime_labels, regime_analysis, hmm_states)
            
            # Step 2: Train models for each regime
            self.logger.info("🔄 Step 2: Training models for each regime...")
            regime_results = self.train_regime_models(regime_data, feature_names)
            
            # Step 3: Save models
            if self.config.save_models:
                self.logger.info("🔄 Step 3: Saving trained models...")
                symbol = kwargs.get('symbol')
                exchange = kwargs.get('exchange')
                timeframe = kwargs.get('timeframe', self.config.timeframe)
                self.save_regime_models(regime_results, symbol, exchange, timeframe)
            
            # Step 4: Evaluate performance
            if self.config.enable_evaluation:
                self.logger.info("🔄 Step 4: Evaluating model performance...")
                evaluation_results = self.evaluate_regime_models(
                    regime_results, X, y, regime_labels, 
                    is_classification=kwargs.get('is_classification', True)
                )
            else:
                evaluation_results = {}
            
            # Create final results
            total_time = time.time() - start_time
            results = self._create_final_results(
                models=regime_results['models'],
                metadata=regime_results['metadata'],
                evaluation_results=evaluation_results,
                training_time=total_time,
                additional_results={'regime_analysis': regime_analysis}
            )
            
            self.training_results = results
            
            # Log summary
            n_models = sum(len(models) for models in regime_results['models'].values())
            self._log_training_summary(results, f"Per-regime {self.config.model_name}", n_models)
            
            return results
            
        except Exception as e:
            return self._handle_training_error(e, "per-regime training")