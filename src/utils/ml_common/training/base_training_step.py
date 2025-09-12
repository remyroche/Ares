"""
Base Training Step

Base class for all training steps with common functionality.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from abc import ABC, abstractmethod

from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import BaseTrainingConfig
from src.utils.ml_common.data_processing.regime_processing import RegimeProcessor
from src.utils.ml_common.data_processing.feature_preparation import FeaturePreparator
from src.utils.ml_common.training.training_utils import TrainingUtils
from src.utils.ml_common.models.model_manager import ModelManager
from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils

logger = system_logger.getChild('BaseTrainingStep')


class BaseTrainingStep(ABC):
    """
    Base class for all training steps with common functionality.
    
    This class provides common functionality that can be inherited by specific
    training modules, reducing code duplication and ensuring consistency.
    """
    
    def __init__(self, config: BaseTrainingConfig):
        """
        Initialize base training step.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        
        # Initialize common components
        self._initialize_common_components()
        
        # Training results
        self.training_results = {}
        
        self.logger.info("✅ Base Training Step initialized")
    
    def _initialize_common_components(self):
        """Initialize common components used by all training steps."""
        # Initialize training utilities
        self.training_utils = TrainingUtils(self.config)
        
        # Initialize data processors
        self.regime_processor = RegimeProcessor()
        self.feature_preparator = FeaturePreparator()
        
        # Initialize model manager
        self.model_manager = ModelManager(
            save_path=self.config.model_save_path,
            save_format=self.config.save_format
        )
        
        # Initialize evaluation utilities
        self.evaluation_utils = EvaluationUtils()
    
    @abstractmethod
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
        Execute training step.
        
        This method must be implemented by subclasses.
        
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
        pass
    
    def analyze_regimes(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze regime distribution and characteristics.
        
        Args:
            regime_labels: Array of regime labels for each sample
            
        Returns:
            Dictionary containing regime analysis results
        """
        return self.regime_processor.analyze_regimes(
            regime_labels=regime_labels,
            min_samples=self.config.min_samples_per_regime,
            enable_regime_merging=self.config.enable_regime_merging,
            regime_merge_threshold=self.config.regime_merge_threshold
        )
    
    def prepare_regime_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        regime_analysis: Dict[str, Any],
        hmm_states: Optional[np.ndarray] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Prepare data for each regime with HMM state integration.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            regime_analysis: Results from regime analysis
            hmm_states: Optional HMM cluster/regime states
            
        Returns:
            Dictionary containing prepared data for each regime
        """
        return self.regime_processor.prepare_regime_data(
            X=X,
            y=y,
            regime_labels=regime_labels,
            regime_analysis=regime_analysis,
            hmm_states=hmm_states,
            min_samples=self.config.min_samples_per_regime,
            enable_data_augmentation=self.config.enable_data_augmentation,
            augmentation_method=self.config.augmentation_method,
            augmentation_ratio=self.config.augmentation_ratio
        )
    
    def prepare_combined_features(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
        hmm_states: Optional[np.ndarray] = None,
        analyst_outputs: Optional[np.ndarray] = None,
        analyst_output_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare combined features with HMM states, analyst outputs, and regime features.
        
        Args:
            X: Input features
            regime_labels: Array of regime labels
            hmm_states: Optional HMM cluster/regime states
            analyst_outputs: Optional analyst model outputs
            analyst_output_names: Names of analyst output features
            feature_names: Names of input features
            
        Returns:
            Tuple of combined features and feature names
        """
        return self.feature_preparator.prepare_combined_features(
            X=X,
            regime_labels=regime_labels,
            hmm_states=hmm_states,
            analyst_outputs=analyst_outputs,
            analyst_output_names=analyst_output_names,
            feature_names=feature_names
        )
    
    def train_models(
        self,
        model_types: List[str],
        X: np.ndarray,
        y: np.ndarray,
        enable_hpo: bool = True,
        search_spaces: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Args:
            model_types: List of model types to train
            X: Input features
            y: Target values
            enable_hpo: Whether to use HPO
            search_spaces: HPO search spaces for each model type
            
        Returns:
            Dictionary containing training results
        """
        return self.training_utils.train_models(
            model_types=model_types,
            X=X,
            y=y,
            enable_hpo=enable_hpo,
            search_spaces=search_spaces
        )
    
    def evaluate_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of trained models
            X: Input features
            y: True target values
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing evaluation results for each model
        """
        return self.training_utils.evaluate_models(
            models=models,
            X=X,
            y=y,
            is_classification=is_classification
        )
    
    def save_models(
        self,
        models: Dict[str, Any],
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> List[str]:
        """
        Save trained models.
        
        Args:
            models: Dictionary of models to save
            model_type: Type of models
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier
            
        Returns:
            List of saved model file paths
        """
        return self.model_manager.save_models(
            models=models,
            model_type=model_type,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            regime=regime
        )
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        model_type: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        timeframe: Optional[str] = None,
        regime: Optional[int] = None
    ) -> str:
        """
        Save model metadata.
        
        Args:
            metadata: Model metadata to save
            model_type: Type of models
            symbol: Optional symbol identifier
            exchange: Optional exchange identifier
            timeframe: Optional timeframe identifier
            regime: Optional regime identifier
            
        Returns:
            Path to saved metadata file
        """
        return self.model_manager.save_metadata(
            metadata=metadata,
            model_type=model_type,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            regime=regime
        )
    
    def get_model_metadata(
        self,
        model: Any,
        model_name: str,
        training_time: float = 0.0,
        optimization_time: float = 0.0,
        samples: int = 0,
        features: int = 0
    ) -> Dict[str, Any]:
        """
        Extract common model metadata.
        
        Args:
            model: Trained model
            model_name: Name of the model
            training_time: Training time in seconds
            optimization_time: Optimization time in seconds
            samples: Number of training samples
            features: Number of features
            
        Returns:
            Dictionary containing model metadata
        """
        return self.model_manager.get_model_metadata(
            model=model,
            model_name=model_name,
            training_time=training_time,
            optimization_time=optimization_time,
            samples=samples,
            features=features
        )
    
    def _create_final_results(
        self,
        models: Dict[str, Any],
        metadata: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        training_time: float,
        additional_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create final results dictionary.
        
        Args:
            models: Trained models
            metadata: Training metadata
            evaluation_results: Evaluation results
            training_time: Total training time
            additional_results: Additional results to include
            
        Returns:
            Dictionary containing final results
        """
        results = {
            'models': models,
            'metadata': metadata,
            'evaluation_results': evaluation_results,
            'training_time': training_time,
            'config': self.config
        }
        
        if additional_results:
            results.update(additional_results)
        
        return results
    
    def _log_training_summary(
        self,
        results: Dict[str, Any],
        model_type: str,
        n_models: int = 0
    ):
        """
        Log training summary.
        
        Args:
            results: Training results
            model_type: Type of models trained
            n_models: Number of models trained
        """
        training_time = results.get('training_time', 0)
        self.logger.info(f"✅ {model_type} training completed in {training_time:.2f}s")
        
        if n_models > 0:
            self.logger.info(f"📊 Models trained: {n_models}")
        
        # Log evaluation results if available
        evaluation_results = results.get('evaluation_results', {})
        if evaluation_results:
            self.logger.info("📊 Evaluation results:")
            for model_name, metrics in evaluation_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    # Log key metrics
                    key_metrics = ['accuracy', 'f1_score', 'r2', 'mse']
                    metric_values = []
                    for metric in key_metrics:
                        if metric in metrics:
                            metric_values.append(f"{metric}={metrics[metric]:.4f}")
                    
                    if metric_values:
                        self.logger.info(f"📊 - {model_name}: {', '.join(metric_values)}")
    
    def _handle_training_error(self, error: Exception, context: str = ""):
        """
        Handle training errors with proper logging.
        
        Args:
            error: Exception that occurred
            context: Additional context about where the error occurred
        """
        error_msg = f"❌ Training error{f' in {context}' if context else ''}: {error}"
        self.logger.error(error_msg)
        
        # Return empty results structure
        return {
            'models': {},
            'metadata': {},
            'evaluation_results': {},
            'training_time': 0,
            'config': self.config,
            'error': str(error)
        }