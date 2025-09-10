"""
Model Training Utilities

This module provides comprehensive model training utilities extracted from training steps
to eliminate code duplication and provide consistent model training across all steps.

Key Features:
- Training data preparation and splitting utilities
- Model metadata generation and tracking
- Training result aggregation and reporting
- Common training patterns and workflows
- Integration with ML Common utilities
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np

# Import ML Common utilities
from src.utils.ml_common import (
    EnhancedModelTrainer,
    ModelEvaluationUtilities,
    DataQualityUtilities,
    MLTrainingSafeguards
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class ModelTrainingUtilities:
    """
    Model training utilities for all training steps.
    
    This provides common model training patterns and utilities
    extracted from multiple training step implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model training utilities."""
        self.config = config or {}
        self.logger = logger.getChild('ModelTrainingUtilities')
        
        # Initialize ML Common utilities
        self.model_trainer = EnhancedModelTrainer(self.config.get('model_training_config', {}))
        self.model_evaluator = ModelEvaluationUtilities(self.config.get('evaluation_config', {}))
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        # Model training configuration
        self.training_config = self.config.get('model_training_config', {})
        
        # Standard model training settings
        self.standard_settings = {
            'enable_confidence_metrics': True,
            'enable_calibration_assessment': True,
            'enable_feature_importance': True,
            'enable_cross_validation': True,
            'enable_model_explanations': True,
            'enable_post_training_hpo': True,
            'cv_folds': 5,
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'enable_class_weights': True,
            'class_weight_config': 'balanced',
            'enable_early_stopping': True,
            'early_stopping_patience': 10,
            'enable_model_persistence': True,
            'model_save_path': 'models'
        }
        
        # Update with user configuration
        self.standard_settings.update(self.training_config)
        
        self.logger.info("🚀 Model Training Utilities initialized")
    
    def prepare_training_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with proper splitting.
        
        Args:
            features: Training features
            targets: Training targets
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Convert to numpy arrays
            X = features.values if hasattr(features, 'values') else features
            y = targets.values if hasattr(targets, 'values') else targets
            
            # Split data
            test_size = self.standard_settings.get('test_size', 0.2)
            random_state = self.standard_settings.get('random_state', 42)
            
            # Handle stratified splitting for classification
            stratify = None
            if len(np.unique(y)) > 1 and len(np.unique(y)) < len(y) * 0.1:  # Classification with reasonable class distribution
                stratify = y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify
            )
            
            self.logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.exception(f"Error preparing training data: {e}")
            raise
    
    def train_basic_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Train basic model with minimal features.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Training result dictionary
        """
        try:
            self.logger.info("Training basic model...")
            
            # Use simple model for basic training
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.standard_settings.get('random_state', 42),
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Basic evaluation
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            return {
                'model': model,
                'evaluation_metrics': {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score
                },
                'feature_importance': dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
            }
            
        except Exception as e:
            self.logger.exception(f"Error training basic model: {e}")
            raise
    
    def train_standard_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Train standard model with cross-validation and basic evaluation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Training result dictionary
        """
        try:
            self.logger.info("Training standard model...")
            
            # Use EnhancedModelTrainer for standard training
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self.standard_settings.get('random_state', 42),
                n_jobs=-1
            )
            
            # Train and evaluate using EnhancedModelTrainer
            training_result = self.model_trainer.train_and_evaluate_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                enable_class_weights=self.standard_settings.get('enable_class_weights', True),
                class_weight_config=self.standard_settings.get('class_weight_config', 'balanced')
            )
            
            return training_result
            
        except Exception as e:
            self.logger.exception(f"Error training standard model: {e}")
            raise
    
    def train_comprehensive_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Train comprehensive model with all features enabled.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Training result dictionary
        """
        try:
            self.logger.info("Training comprehensive model...")
            
            # Use EnhancedModelTrainer with all features enabled
            from sklearn.ensemble import RandomForestClassifier
            
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.standard_settings.get('random_state', 42),
                n_jobs=-1
            )
            
            # Train and evaluate using EnhancedModelTrainer with all features
            training_result = self.model_trainer.train_and_evaluate_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
                enable_class_weights=self.standard_settings.get('enable_class_weights', True),
                class_weight_config=self.standard_settings.get('class_weight_config', 'balanced')
            )
            
            return training_result
            
        except Exception as e:
            self.logger.exception(f"Error training comprehensive model: {e}")
            raise
    
    def generate_training_metadata(self, features: pd.DataFrame, targets: pd.Series, 
                                 training_result: Dict[str, Any], model_type: str, model_name: str) -> Dict[str, Any]:
        """
        Generate metadata about model training.
        
        Args:
            features: Training features
            targets: Training targets
            training_result: Training result dictionary
            model_type: Type of training
            model_name: Name of the model
            
        Returns:
            Training metadata dictionary
        """
        try:
            metadata = {
                'model_type': model_type,
                'model_name': model_name,
                'training_timestamp': datetime.now().isoformat(),
                'data_shape': {
                    'features_shape': features.shape,
                    'targets_shape': targets.shape,
                    'feature_names': list(features.columns) if hasattr(features, 'columns') else None
                },
                'training_settings': self.standard_settings,
                'model_info': {
                    'model_type': type(training_result.get('model', None)).__name__,
                    'model_params': getattr(training_result.get('model', None), 'get_params', lambda: {})()
                }
            }
            
            # Add evaluation metrics summary
            if 'evaluation_metrics' in training_result:
                eval_metrics = training_result['evaluation_metrics']
                metadata['evaluation_summary'] = {
                    'primary_metric': eval_metrics.get('accuracy', eval_metrics.get('test_accuracy', 0)),
                    'metrics_count': len(eval_metrics),
                    'available_metrics': list(eval_metrics.keys())
                }
            
            # Add feature importance summary
            if 'feature_importance' in training_result and training_result['feature_importance']:
                importance_dict = training_result['feature_importance']
                if isinstance(importance_dict, dict):
                    metadata['feature_importance_summary'] = {
                        'top_features': sorted(
                            importance_dict.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:10],
                        'total_features': len(importance_dict),
                        'max_importance': max(importance_dict.values()) if importance_dict else 0,
                        'min_importance': min(importance_dict.values()) if importance_dict else 0
                    }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating training metadata: {e}")
            return {'error': str(e)}
    
    def train_model(self, features: pd.DataFrame, targets: pd.Series, 
                   model_type: str = 'comprehensive', model_name: str = 'default_model') -> Dict[str, Any]:
        """
        Train model using unified approach.
        
        Args:
            features: Training features
            targets: Training targets
            model_type: Type of training ('basic', 'standard', 'comprehensive')
            model_name: Name of the model
            
        Returns:
            Model training result
        """
        try:
            self.logger.info(f"🤖 Starting {model_type} model training for '{model_name}'...")
            
            # Validate input data
            features_validation = self.data_quality.analyze_data_quality(features)
            targets_validation = self.data_quality.analyze_data_quality(targets)
            
            if not features_validation['passed']:
                self.logger.warning(f"Features validation issues: {features_validation.get('warnings', [])}")
            
            if not targets_validation['passed']:
                self.logger.warning(f"Targets validation issues: {targets_validation.get('warnings', [])}")
            
            # Prepare data for training
            X_train, X_test, y_train, y_test = self.prepare_training_data(features, targets)
            
            # Train model based on type
            if model_type == 'basic':
                training_result = self.train_basic_model(X_train, y_train, X_test, y_test, model_name)
            elif model_type == 'standard':
                training_result = self.train_standard_model(X_train, y_train, X_test, y_test, model_name)
            elif model_type == 'comprehensive':
                training_result = self.train_comprehensive_model(X_train, y_train, X_test, y_test, model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Generate training metadata
            training_metadata = self.generate_training_metadata(
                features, targets, training_result, model_type, model_name
            )
            
            return {
                'model': training_result.get('model'),
                'evaluation_metrics': training_result.get('evaluation_metrics', {}),
                'confidence_metrics': training_result.get('confidence_metrics', {}),
                'feature_importance': training_result.get('feature_importance', {}),
                'training_metadata': training_metadata,
                'features_validation': features_validation,
                'targets_validation': targets_validation,
                'model_type': model_type,
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.exception(f"Error training model: {e}")
            raise
    
    def get_model_training_summary(self) -> Dict[str, Any]:
        """Get summary of model training capabilities."""
        return {
            'config': self.config,
            'standard_settings': self.standard_settings,
            'model_trainer_info': {
                'trainer_type': 'EnhancedModelTrainer',
                'available_features': [
                    'confidence_metrics',
                    'calibration_assessment',
                    'feature_importance',
                    'cross_validation',
                    'model_explanations',
                    'post_training_hpo'
                ]
            },
            'timestamp': datetime.now().isoformat()
        }


# Global instance for easy access
_global_training_utilities = None

def get_model_training_utilities(config: Optional[Dict[str, Any]] = None) -> ModelTrainingUtilities:
    """Get model training utilities instance."""
    return ModelTrainingUtilities(config)


# Convenience functions
def prepare_training_data(features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training data using utilities."""
    utils = get_model_training_utilities()
    return utils.prepare_training_data(features, targets)


def train_model(features: pd.DataFrame, targets: pd.Series, 
               model_type: str = 'comprehensive', model_name: str = 'default_model') -> Dict[str, Any]:
    """Train model using utilities."""
    utils = get_model_training_utilities()
    return utils.train_model(features, targets, model_type, model_name)


def generate_training_metadata(features: pd.DataFrame, targets: pd.Series, 
                             training_result: Dict[str, Any], model_type: str, model_name: str) -> Dict[str, Any]:
    """Generate training metadata using utilities."""
    utils = get_model_training_utilities()
    return utils.generate_training_metadata(features, targets, training_result, model_type, model_name)


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    
    # Create features with some signal
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some signal to first 10 features
    for i in range(10):
        features[f'feature_{i}'] += np.random.randn(n_samples) * 0.5
    
    # Create targets based on first 10 features
    targets = pd.Series(
        (features.iloc[:, :10].sum(axis=1) > 0).astype(int),
        name='target'
    )
    
    # Test model training utilities
    utils = ModelTrainingUtilities()
    
    print("=== Training Data Preparation ===")
    X_train, X_test, y_train, y_test = utils.prepare_training_data(features, targets)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    print("\n=== Basic Model Training ===")
    basic_result = utils.train_model(features, targets, 'basic', 'basic_model')
    print(f"Basic model accuracy: {basic_result['evaluation_metrics'].get('test_accuracy', 0):.3f}")
    
    print("\n=== Standard Model Training ===")
    standard_result = utils.train_model(features, targets, 'standard', 'standard_model')
    print(f"Standard model accuracy: {standard_result['evaluation_metrics'].get('test_accuracy', 0):.3f}")
    
    print("\n=== Comprehensive Model Training ===")
    comprehensive_result = utils.train_model(features, targets, 'comprehensive', 'comprehensive_model')
    print(f"Comprehensive model accuracy: {comprehensive_result['evaluation_metrics'].get('test_accuracy', 0):.3f}")
    
    print("\n=== Training Metadata ===")
    metadata = utils.generate_training_metadata(features, targets, comprehensive_result, 'comprehensive', 'comprehensive_model')
    print(f"Model type: {metadata['model_type']}")
    print(f"Training timestamp: {metadata['training_timestamp']}")
    print(f"Data shape: {metadata['data_shape']['features_shape']}")
    
    print("\n=== Training Summary ===")
    summary = utils.get_model_training_summary()
    print(f"Available features: {summary['model_trainer_info']['available_features']}")
    print(f"Standard settings: {list(summary['standard_settings'].keys())}")