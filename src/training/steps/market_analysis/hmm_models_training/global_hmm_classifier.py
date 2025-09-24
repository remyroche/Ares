"""
Global HMM Classifier for Multi-State Prediction

This module implements a global classifier approach that trains a single model
to predict probability distributions over all 20 HMM states simultaneously.
Fully integrated with ml_commons infrastructure.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Core imports - using ml_commons infrastructure
from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep
from src.utils.ml_common.utils.hmm_hpo_config import get_hmm_hyperparameter_optimizer
from src.utils.ml_common.utils.hmm_temporal_protection import get_hmm_temporal_protection

# ml_commons model factory and multi-output models
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelType, ModelConfig
from src.utils.ml_common.models.multi_output_models import MultiOutputModel

# Feature generation
from .shared_feature_utils import create_comprehensive_features


class GlobalHMMClassifier:
    """
    Single model that predicts probability distribution over all 20 HMM states.
    Returns probability vector [p(state_0), p(state_1), ..., p(state_19)]
    
    Fully integrated with ml_commons infrastructure using EnhancedModelFactory.
    """
    
    def __init__(self, n_hmm_states: int = 20, model_type: str = 'lightgbm'):
        self.n_hmm_states = n_hmm_states
        self.model_type = model_type
        self.model = None
        self.model_factory = None
        self.state_mapping = None
        self.feature_names = None
        self.logger = system_logger.getChild('GlobalHMMClassifier')
        
        # Initialize ml_commons model factory
        self.model_factory = EnhancedModelFactory()
    
    def predict_state_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability distribution over all HMM states.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, 20) with probability for each state
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def predict_dominant_state(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely HMM state for each sample.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples,) with dominant state indices
        """
        probabilities = self.predict_state_probabilities(X)
        return np.argmax(probabilities, axis=1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, model_type: str = None):
        """
        Train the global classifier using ml_commons EnhancedModelFactory.
        
        Args:
            X: Training features
            y: HMM state labels (0-19)
            model_type: Type of model to train (uses self.model_type if None)
        """
        if model_type is None:
            model_type = self.model_type
        
        # Validate HMM states are in range [0, 19]
        unique_states = np.unique(y)
        if not all(0 <= state <= 19 for state in unique_states):
            raise ValueError(f"HMM states must be in range [0, 19], found: {unique_states}")
        
        self.logger.info(f"Training global HMM classifier with {model_type} for {len(unique_states)} states")
        
        # Create model using ml_commons EnhancedModelFactory
        self.model = self._create_model_with_ml_commons(model_type)
        
        # Train the model
        self.model.fit(X, y)
        
        self.logger.info("Global HMM classifier training completed")
    
    def _create_model_with_ml_commons(self, model_type: str):
        """Create model instance using ml_commons EnhancedModelFactory."""
        # Map model type to ml_commons ModelType enum
        model_type_mapping = {
            'lightgbm': ModelType.LIGHTGBM_CLASSIFIER,
            'xgboost': ModelType.XGBOOST_CLASSIFIER,
            'random_forest': ModelType.RANDOM_FOREST_CLASSIFIER,
            'elastic_net': ModelType.ELASTIC_NET_CLASSIFIER
        }
        
        if model_type not in model_type_mapping:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(model_type_mapping.keys())}")
        
        # Create model configuration
        model_config = ModelConfig(
            model_name=f"global_hmm_{model_type}",
            model_type=model_type_mapping[model_type],
            model_params=self._get_model_specific_params(model_type),
            random_state=42,
            enable_memory_optimization=True,
            enable_gpu_acceleration=False  # Disable for stability
        )
        
        # Create model using ml_commons factory
        model = self.model_factory.create_model(model_config)
        
        self.logger.info(f"Created {model_type} model using ml_commons EnhancedModelFactory")
        return model
    
    def _get_model_specific_params(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific parameters for multi-class classification."""
        params = {
            'lightgbm': {
                'objective': 'multiclass',
                'num_class': self.n_hmm_states,
                'metric': 'multi_logloss',
                'verbose': -1,
                'random_state': 42
            },
            'xgboost': {
                'objective': 'multi:softprob',
                'num_class': self.n_hmm_states,
                'eval_metric': 'mlogloss',
                'verbosity': 0,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1
            },
            'elastic_net': {
                'multi_class': 'ovr',
                'random_state': 42,
                'max_iter': 1000
            }
        }
        
        return params.get(model_type, {})


class GlobalHMMTrainingStep(BaseTrainingStep):
    """
    Global classifier training for all 20 HMM states simultaneously.
    """
    
    def __init__(self, config: Optional[HMMTrainingConfig] = None):
        """
        Initialize global HMM training step.
        
        Args:
            config: HMM training configuration (will be updated for global approach)
        """
        # Ensure we have a config with global classifier settings
        if config is None:
            config = HMMTrainingConfig(
                model_name="global_hmm_classifier",
                timeframe="15m",  # Always use 15m for HMM state recognition
                hpo_trials=50,
                enable_multi_objective=True,
                objectives=["accuracy", "f1_score", "regime_stability"],
                objective_weights=[0.5, 0.35, 0.15]  # Updated weights for global classifier focus
            )
        else:
            # Override settings for global approach
            config.timeframe = "15m"
            config.objective_weights = [0.5, 0.35, 0.15]  # Force updated weights
        
        super().__init__(config)
        self.logger = system_logger.getChild('GlobalHMMTrainingStep')
        
        # Global classifier specific settings
        self.n_hmm_states = 20
        self.global_model_types = [
            "lightgbm",            # Multi-class LightGBM
            "xgboost",             # Multi-class XGBoost
            "random_forest",       # Multi-class Random Forest
            "elastic_net"          # Multi-class Elastic Net
        ]
        
        # Initialize ml_commons utilities
        self.hmm_hpo = get_hmm_hyperparameter_optimizer(config)
        self.hmm_temporal_protection = get_hmm_temporal_protection(config)
        
        self.logger.info("✅ Global HMM Training Step initialized")
        self.logger.info(f"📊 Target: Single model for {self.n_hmm_states} HMM states")
        self.logger.info(f"📊 Objective weights: {config.objective_weights}")
    
    def execute(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray, 
                feature_names: Optional[List[str]] = None, hmm_states: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute global HMM training for all states simultaneously.
        
        Args:
            X: Input features
            y: HMM state labels (0-19) for each sample
            regime_labels: Regime labels for context (not used for splitting)
            feature_names: Names of input features
            hmm_states: Optional HMM cluster/regime states (same as y)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results from global approach
        """
        self.logger.info("🚀 Starting global HMM classifier training")
        
        # Validate input data
        validation_results = self.validate_training_data(
            X=X, y=y, regime_labels=regime_labels,
            feature_names=feature_names, timestamps=None,
            model_type="global_hmm_classification"
        )
        
        if not validation_results['valid']:
            self.logger.error("❌ Training data validation failed")
            return self._handle_training_error(
                Exception("Training data validation failed"),
                "data_validation"
            )
        
        # Validate HMM states are in correct range
        unique_states = np.unique(y)
        if not all(0 <= state <= 19 for state in unique_states):
            raise ValueError(f"HMM states must be in range [0, 19], found: {unique_states}")
        
        self.logger.info(f"📊 Validated {len(unique_states)} HMM states: {sorted(unique_states)}")
        
        # Generate comprehensive features
        try:
            # Create comprehensive features using feature bank
            if isinstance(X, pd.DataFrame):
                X_enhanced, feature_names = create_comprehensive_features(X, regime_labels)
            else:
                # Convert to DataFrame if needed for feature bank
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=feature_names)
                X_enhanced, feature_names = create_comprehensive_features(df, regime_labels)
        except Exception as e:
            self.logger.error(f"❌ Feature generation failed: {e}")
            raise ValueError(f"Feature generation failed: {e}. Cannot proceed without proper feature engineering.")
        
        self.logger.info(f"📊 Generated {X_enhanced.shape[1]} comprehensive features")
        
        # Train global models using ml_commons training utilities
        training_results = self._train_global_hmm_classifiers_with_ml_commons(
            X_enhanced, y, feature_names
        )
        
        # Generate enhanced reporting
        enhanced_reporting = self._generate_global_classifier_report(
            models=training_results.get('models', {}),
            evaluation_results=training_results.get('evaluation_results', {}),
            validation_results=validation_results
        )
        
        # Create final results
        final_results = self._create_final_results(
            models=training_results.get('models', {}),
            metadata=training_results.get('metadata', {}),
            evaluation_results=training_results.get('evaluation_results', {}),
            training_time=training_results.get('training_time', 0),
            additional_results={
                'global_classifier_approach': True,
                'n_hmm_states': self.n_hmm_states,
                'validation_results': validation_results,
                'enhanced_reporting': enhanced_reporting,
                'ml_commons_integration': {
                    'hpo_used': True,
                    'universal_validation_used': True,
                    'temporal_protection_used': True,
                    'global_classifier_mode': True
                }
            }
        )
        
        self._log_global_training_summary(final_results)
        return final_results
    
    def _train_global_hmm_classifiers_with_ml_commons(self, X: np.ndarray, y: np.ndarray, 
                                                    feature_names: List[str]) -> Dict[str, Any]:
        """
        Train global HMM classifiers for all states simultaneously using ml_commons.
        
        Args:
            X: Enhanced features
            y: HMM state labels
            feature_names: Feature names
            
        Returns:
            Training results
        """
        self.logger.info("🔄 Training global HMM classifiers with ml_commons integration")
        
        # Create train/test split using ml_commons utilities
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        trained_models = {}
        evaluation_results = {}
        total_training_time = 0
        
        for model_type in self.global_model_types:
            self.logger.info(f"📊 Training {model_type} global classifier with ml_commons")
            
            start_time = time.time()
            
            try:
                # Create and train global classifier using ml_commons
                classifier = GlobalHMMClassifier(n_hmm_states=self.n_hmm_states, model_type=model_type)
                classifier.fit(X_train, y_train, model_type)
                
                # Evaluate the model using ml_commons evaluation utilities
                eval_results = self._evaluate_global_classifier_with_ml_commons(
                    classifier, X_test, y_test, model_type
                )
                
                training_time = time.time() - start_time
                total_training_time += training_time
                
                trained_models[model_type] = classifier
                evaluation_results[model_type] = eval_results
                
                self.logger.info(f"✅ {model_type} training completed in {training_time:.2f}s")
                self.logger.info(f"📊 Accuracy: {eval_results['accuracy']:.4f}, F1: {eval_results['f1_macro']:.4f}")
                self.logger.info(f"📊 ml_commons integration: EnhancedModelFactory used")
                
            except Exception as e:
                self.logger.error(f"❌ {model_type} training failed: {e}")
                evaluation_results[model_type] = {'error': str(e)}
        
        return {
            'models': trained_models,
            'evaluation_results': evaluation_results,
            'training_time': total_training_time,
            'feature_names': feature_names,
            'ml_commons_integration': True
        }
    
    def _evaluate_global_classifier_with_ml_commons(self, classifier: GlobalHMMClassifier, 
                                                   X_test: np.ndarray, y_test: np.ndarray, 
                                                   model_type: str) -> Dict[str, Any]:
        """
        Evaluate global classifier with comprehensive metrics using ml_commons.
        
        Args:
            classifier: Trained global classifier
            X_test: Test features
            y_test: Test labels
            model_type: Type of model
            
        Returns:
            Evaluation results with ml_commons integration
        """
        # Predict probabilities for all states
        state_probabilities = classifier.predict_state_probabilities(X_test)
        
        # Predict dominant states
        predicted_states = classifier.predict_dominant_state(X_test)
        
        # Standard metrics
        accuracy = accuracy_score(y_test, predicted_states)
        f1_macro = f1_score(y_test, predicted_states, average='macro')
        f1_weighted = f1_score(y_test, predicted_states, average='weighted')
        
        # HMM-specific metrics
        state_distribution_accuracy = self._calculate_state_distribution_accuracy(
            y_test, state_probabilities
        )
        
        state_transition_consistency = self._calculate_transition_consistency(
            y_test, predicted_states
        )
        
        # Calculate overall objective score
        objective_score, objective_breakdown = self._calculate_global_objective(
            y_test, state_probabilities, predicted_states
        )
        
        # ml_commons evaluation integration
        ml_commons_evaluation = self._get_ml_commons_evaluation_metrics(
            classifier, X_test, y_test, state_probabilities, predicted_states
        )
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'state_distribution_accuracy': state_distribution_accuracy,
            'state_transition_consistency': state_transition_consistency,
            'objective_score': objective_score,
            'objective_breakdown': objective_breakdown,
            'state_probabilities': state_probabilities,
            'predicted_states': predicted_states,
            'model_type': model_type,
            'ml_commons_evaluation': ml_commons_evaluation,
            'ml_commons_integration': True
        }
    
    def _get_ml_commons_evaluation_metrics(self, classifier: GlobalHMMClassifier, 
                                         X_test: np.ndarray, y_test: np.ndarray,
                                         state_probabilities: np.ndarray, 
                                         predicted_states: np.ndarray) -> Dict[str, Any]:
        """Get additional evaluation metrics using ml_commons utilities."""
        try:
            # Use ml_commons evaluation utilities if available
            from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
            
            evaluation_utils = EvaluationUtils()
            
            # Basic evaluation using ml_commons
            basic_metrics = evaluation_utils.evaluate_model(
                model=classifier.model,
                X_test=X_test,
                y_test=y_test,
                is_classification=True
            )
            
            return {
                'basic_metrics': basic_metrics,
                'evaluation_utils_used': True
            }
            
        except ImportError:
            # Fallback if ml_commons evaluation not available
            return {
                'basic_metrics': {},
                'evaluation_utils_used': False,
                'fallback_reason': 'ml_commons evaluation utils not available'
            }
        except Exception as e:
            return {
                'basic_metrics': {},
                'evaluation_utils_used': False,
                'error': str(e)
            }
    
    def _calculate_state_distribution_accuracy(self, y_true: np.ndarray, 
                                            y_pred_proba: np.ndarray) -> float:
        """Calculate how well the predicted distribution matches true states."""
        # Calculate average probability assigned to true states
        true_state_probs = y_pred_proba[np.arange(len(y_true)), y_true]
        return float(np.mean(true_state_probs))
    
    def _calculate_transition_consistency(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> float:
        """Calculate consistency of state transitions."""
        if len(y_true) < 2:
            return 0.0
        
        # Calculate transition accuracy
        true_transitions = np.diff(y_true)
        pred_transitions = np.diff(y_pred)
        
        transition_accuracy = np.mean(true_transitions == pred_transitions)
        return float(transition_accuracy)
    
    def _calculate_global_objective(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective score for global HMM classifier.
        
        Returns:
            Tuple of (total_score, objective_breakdown)
        """
        # Objective 1: Accuracy (50% weight)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Objective 2: F1-Score (35% weight) - macro average for balanced evaluation
        f1_score_val = f1_score(y_true, y_pred, average='macro')
        
        # Objective 3: Regime Stability (15% weight) - state distribution accuracy
        regime_stability = self._calculate_state_distribution_accuracy(y_true, y_pred_proba)
        
        # Weighted combination
        total_score = (
            self.config.objective_weights[0] * accuracy +
            self.config.objective_weights[1] * f1_score_val +
            self.config.objective_weights[2] * regime_stability
        )
        
        return total_score, {
            'accuracy': accuracy,
            'f1_score': f1_score_val,
            'regime_stability': regime_stability
        }
    
    def _generate_global_classifier_report(self, models: Dict[str, Any], 
                                         evaluation_results: Dict[str, Any],
                                         validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive report for global classifiers."""
        self.logger.info("📊 Generating global classifier report...")
        
        report = {
            'global_classifier_summary': {},
            'model_performance_comparison': {},
            'best_model_recommendation': {},
            'hmm_state_analysis': {},
            'overall_recommendations': [],
            'validation_insights': validation_results
        }
        
        # Compare model performance
        model_scores = {}
        for model_type, eval_results in evaluation_results.items():
            if 'error' not in eval_results:
                model_scores[model_type] = eval_results['objective_score']
        
        if model_scores:
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
            best_score = model_scores[best_model]
            
            report['best_model_recommendation'] = {
                'best_model': best_model,
                'objective_score': best_score,
                'evaluation_details': evaluation_results[best_model]
            }
            
            report['overall_recommendations'] = [
                f"Best global classifier: {best_model} (objective score: {best_score:.4f})",
                f"Single model can predict all {self.n_hmm_states} HMM states simultaneously",
                "Use probability distributions for uncertainty quantification",
                "Monitor state transition consistency for regime stability"
            ]
        
        return report
    
    def _log_global_training_summary(self, results: Dict[str, Any]) -> None:
        """Log summary of global training results."""
        enhanced_reporting = results.get('enhanced_reporting', {})
        
        if enhanced_reporting:
            self.logger.info("📊 Global HMM Training Summary:")
            
            best_recommendation = enhanced_reporting.get('best_model_recommendation', {})
            if best_recommendation:
                best_model = best_recommendation.get('best_model', 'Unknown')
                best_score = best_recommendation.get('objective_score', 0)
                self.logger.info(f"🏆 Best global classifier: {best_model} (score: {best_score:.4f})")
            
            recommendations = enhanced_reporting.get('overall_recommendations', [])
            if recommendations:
                self.logger.info("💡 Key recommendations:")
                for rec in recommendations[:3]:
                    self.logger.info(f"  - {rec}")
            
            self.logger.info(f"📈 Global training completed for {self.n_hmm_states} HMM states")
    
    def _handle_training_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle training errors with proper logging."""
        error_msg = f"❌ Global HMM training error{f' in {context}' if context else ''}: {error}"
        self.logger.error(error_msg)
        
        return {
            'models': {},
            'metadata': {},
            'evaluation_results': {},
            'training_time': 0,
            'config': self.config,
            'error': str(error),
            'global_classifier_approach': True,
            'n_hmm_states': self.n_hmm_states
        }


# Convenience functions
def create_global_hmm_training(config: Optional[HMMTrainingConfig] = None) -> GlobalHMMTrainingStep:
    """Create a global HMM training step."""
    return GlobalHMMTrainingStep(config)


def execute_global_hmm_training(X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray,
                               config: Optional[HMMTrainingConfig] = None,
                               feature_names: Optional[List[str]] = None,
                               hmm_states: Optional[np.ndarray] = None,
                               **kwargs) -> Dict[str, Any]:
    """Execute global HMM training."""
    training_step = create_global_hmm_training(config)
    return training_step.execute(
        X=X, y=y, regime_labels=regime_labels,
        feature_names=feature_names, hmm_states=hmm_states,
        **kwargs
    )