"""
Unified Model Training Infrastructure

This module provides unified model training across all training steps using
EnhancedModelTrainer from ml_common, replacing custom training implementations.

Key Features:
- Unified model training using EnhancedModelTrainer
- Standardized model evaluation using ModelEvaluationUtilities
- Automatic confidence metrics and calibration assessment
- Feature importance analysis and model explanations
- Comprehensive error handling and logging
- Integration with ML Common utilities
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import pandas as pd
import numpy as np

# Import new simplified infrastructure
from .simplified_pipeline_infrastructure import (
    create_simple_step_function,
    create_data_processing_step_function
)

# Import standardized validation
from .standardized_config_validation import (
    validate_config,
    validate_and_fix_config
)

# Import unified data quality
from .unified_data_quality import (
    validate_data_quality,
    clean_data,
    generate_quality_report
)

# Import ML Common utilities
from src.utils.ml_common import (
    EnhancedModelTrainer,
    ModelEvaluationUtilities,
    DataQualityUtilities,
    MLTrainingSafeguards,
    FeatureSelectionFramework
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class UnifiedModelTrainingManager:
    """
    Unified model training manager for all training steps.
    
    This replaces custom model training implementations with a unified
    approach using EnhancedModelTrainer from ml_common.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified model training manager."""
        self.config = validate_and_fix_config(config, 'model_training')
        self.logger = logger.getChild('UnifiedModelTrainingManager')
        
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
        
        self.logger.info("🚀 Unified Model Training Manager initialized")
    
    async def train_model(self, features: pd.DataFrame, targets: pd.Series, 
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
            features_validation = validate_data_quality(features, 'features', 'comprehensive')
            targets_validation = validate_data_quality(targets, 'targets', 'standard')
            
            if not features_validation['passed']:
                self.logger.warning(f"Features validation issues: {features_validation['errors']}")
            
            if not targets_validation['passed']:
                self.logger.warning(f"Targets validation issues: {targets_validation['errors']}")
            
            # Prepare data for training
            X_train, X_test, y_train, y_test = await self._prepare_training_data(features, targets)
            
            # Train model based on type
            if model_type == 'basic':
                training_result = await self._train_basic_model(X_train, y_train, X_test, y_test, model_name)
            elif model_type == 'standard':
                training_result = await self._train_standard_model(X_train, y_train, X_test, y_test, model_name)
            elif model_type == 'comprehensive':
                training_result = await self._train_comprehensive_model(X_train, y_train, X_test, y_test, model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Generate training metadata
            training_metadata = self._generate_training_metadata(
                features, targets, training_result, model_type, model_name
            )
            
            # Generate quality report
            quality_report = generate_quality_report(training_result.get('model'), 'trained_model')
            
            return {
                'model': training_result.get('model'),
                'evaluation_metrics': training_result.get('evaluation_metrics', {}),
                'confidence_metrics': training_result.get('confidence_metrics', {}),
                'feature_importance': training_result.get('feature_importance', {}),
                'training_metadata': training_metadata,
                'features_validation': features_validation,
                'targets_validation': targets_validation,
                'quality_report': quality_report,
                'model_type': model_type,
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.exception(f"Error training model: {e}")
            raise
    
    async def _prepare_training_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with proper splitting."""
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
    
    async def _train_basic_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                               X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Train basic model with minimal features."""
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
    
    async def _train_standard_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Train standard model with cross-validation and basic evaluation."""
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
    
    async def _train_comprehensive_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                       X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Train comprehensive model with all features enabled."""
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
    
    def _generate_training_metadata(self, features: pd.DataFrame, targets: pd.Series, 
                                  training_result: Dict[str, Any], model_type: str, model_name: str) -> Dict[str, Any]:
        """Generate metadata about model training."""
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


# Simplified model training step functions
async def unified_model_training_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified model training logic using EnhancedModelTrainer.
    
    Args:
        config: Configuration dictionary
        pipeline_state: Current pipeline state
        
    Returns:
        Model training result
    """
    logger.info("🤖 Starting unified model training...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model training")
        
        # Initialize unified model training manager
        training_manager = UnifiedModelTrainingManager(config)
        
        # Determine model type and name from configuration
        model_type = config.get('model_type', 'comprehensive')
        model_name = config.get('model_name', 'unified_model')
        
        # Train model
        result = await training_manager.train_model(features, targets, model_type, model_name)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in unified model training: {e}")
        raise


async def basic_model_training_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Basic model training logic (simple model with minimal features)."""
    logger.info("🤖 Starting basic model training...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model training")
        
        # Initialize unified model training manager
        training_manager = UnifiedModelTrainingManager(config)
        
        # Train basic model
        result = await training_manager.train_model(features, targets, 'basic', 'basic_model')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in basic model training: {e}")
        raise


async def standard_model_training_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Standard model training logic (cross-validation and basic evaluation)."""
    logger.info("🤖 Starting standard model training...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model training")
        
        # Initialize unified model training manager
        training_manager = UnifiedModelTrainingManager(config)
        
        # Train standard model
        result = await training_manager.train_model(features, targets, 'standard', 'standard_model')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in standard model training: {e}")
        raise


async def comprehensive_model_training_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive model training logic (all features enabled)."""
    logger.info("🤖 Starting comprehensive model training...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model training")
        
        # Initialize unified model training manager
        training_manager = UnifiedModelTrainingManager(config)
        
        # Train comprehensive model
        result = await training_manager.train_model(features, targets, 'comprehensive', 'comprehensive_model')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in comprehensive model training: {e}")
        raise


# Create step functions
unified_model_training = create_simple_step_function("unified_model_training", unified_model_training_logic)
basic_model_training = create_simple_step_function("basic_model_training", basic_model_training_logic)
standard_model_training = create_simple_step_function("standard_model_training", standard_model_training_logic)
comprehensive_model_training = create_simple_step_function("comprehensive_model_training", comprehensive_model_training_logic)


class SimplifiedModelTraining:
    """
    Simplified model training using unified infrastructure.
    
    This replaces custom model training implementations with a unified
    approach using EnhancedModelTrainer from ml_common.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified model training."""
        self.config = validate_and_fix_config(config, 'model_training')
        self.logger = logger.getChild('SimplifiedModelTraining')
        
        # Initialize unified model training manager
        self.training_manager = UnifiedModelTrainingManager(self.config)
        
        self.logger.info("🚀 Simplified Model Training initialized")
    
    async def train_model(self, features: pd.DataFrame, targets: pd.Series, 
                         model_type: str = 'comprehensive', model_name: str = 'default_model') -> Dict[str, Any]:
        """
        Train model using unified approach.
        
        Args:
            features: Training features
            targets: Training targets
            model_type: Type of training
            model_name: Name of the model
            
        Returns:
            Model training result
        """
        try:
            self.logger.info(f"🚀 Training {model_type} model '{model_name}'...")
            
            # Train model
            result = await self.training_manager.train_model(features, targets, model_type, model_name)
            
            self.logger.info(f"✅ Model training completed: {result['training_metadata']['model_name']}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Model training error: {e}")
            raise
    
    def get_model_training_summary(self) -> Dict[str, Any]:
        """Get summary of model training capabilities."""
        return self.training_manager.get_model_training_summary()


# Backward compatibility wrappers
class HMMBasedTraining(SimplifiedModelTraining):
    """Backward compatibility wrapper for HMMBasedTraining."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for HMMBasedTraining")


# Import from consolidated analyst and tactician training
from .consolidated_analyst_tactician_training import (
    AnalystEnhancement,
    TacticianSpecialistTraining
)


# Example usage and testing
async def example_model_training():
    """Example of using the unified model training."""
    
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
    
    # Configuration for different training types
    configs = [
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'model_type': 'basic',
            'model_name': 'basic_model',
            'model_training_config': {
                'enable_confidence_metrics': False,
                'enable_calibration_assessment': False,
                'enable_feature_importance': True,
                'enable_cross_validation': False
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'model_type': 'standard',
            'model_name': 'standard_model',
            'model_training_config': {
                'enable_confidence_metrics': True,
                'enable_calibration_assessment': True,
                'enable_feature_importance': True,
                'enable_cross_validation': True,
                'cv_folds': 5
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'model_type': 'comprehensive',
            'model_name': 'comprehensive_model',
            'model_training_config': {
                'enable_confidence_metrics': True,
                'enable_calibration_assessment': True,
                'enable_feature_importance': True,
                'enable_cross_validation': True,
                'enable_model_explanations': True,
                'enable_post_training_hpo': True,
                'cv_folds': 5
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Model Training Type {i+1}: {config['model_type']} ===")
        
        # Create simplified model training
        model_trainer = SimplifiedModelTraining(config)
        
        # Train model
        result = await model_trainer.train_model(features, targets, config['model_type'], config['model_name'])
        
        # Get summary
        summary = model_trainer.get_model_training_summary()
        
        print(f"Model type: {result['model_type']}")
        print(f"Model name: {result['model_name']}")
        print(f"Training timestamp: {result['training_metadata']['training_timestamp']}")
        print(f"Data shape: {result['training_metadata']['data_shape']['features_shape']}")
        
        # Show evaluation metrics
        if 'evaluation_metrics' in result:
            eval_metrics = result['evaluation_metrics']
            print(f"Evaluation metrics: {list(eval_metrics.keys())}")
            if 'accuracy' in eval_metrics:
                print(f"Accuracy: {eval_metrics['accuracy']:.3f}")
        
        # Show feature importance
        if 'feature_importance' in result and result['feature_importance']:
            importance = result['feature_importance']
            if isinstance(importance, dict):
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"Top 5 features: {top_features}")
        
        results.append((result, summary))
    
    return results


# Main execution
async def main():
    """Main execution function."""
    try:
        results = await example_model_training()
        print("✅ Model training example completed successfully")
        return results
    except Exception as e:
        logger.exception(f"Model training example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())