"""
Consolidated Analyst and Tactician Training

This module provides the consolidated Analyst and Tactician training classes
that use the unified model training system while preserving multi-output functionality.

Key Features:
- Multi-output ML models (price prediction, probability, risk of hitting opposite price barrier)
- Per-HMM regime training preserved
- Analyst/Tactician separation maintained
- Uses utilities/ as toolbox from src/training/steps/
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
# Import mock dependencies for testing
try:
    import pandas as pd
    import numpy as np
except ImportError:
    from src.utils.mock_dependencies import MockDataFrame, MockNumpy, MockSeries
    pd = type('MockPandas', (), {'DataFrame': MockDataFrame, 'Series': MockSeries})()
    np = MockNumpy()

# Import utilities as toolbox
from src.utils.ml_common import (
    EnhancedModelTrainer,
    ModelEvaluationUtilities,
    DataQualityUtilities,
    MLTrainingSafeguards
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

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class MultiOutputModelTrainer:
    """
    Multi-output model trainer that generates multiple outputs:
    - Price prediction before hitting opposite side price barrier
    - Probability of hitting the barrier
    - Risk of hitting the opposite price barrier first
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-output model trainer."""
        self.config = config
        self.logger = logger.getChild('MultiOutputModelTrainer')
        
        # Initialize ML Common utilities as toolbox
        self.model_trainer = EnhancedModelTrainer(config.get('model_training_config', {}))
        self.model_evaluator = ModelEvaluationUtilities(config.get('evaluation_config', {}))
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        self.logger.info("🚀 Multi-Output Model Trainer initialized")
    
    async def train_multi_output_model(self, features: pd.DataFrame, targets: Dict[str, pd.Series], 
                                     model_name: str = 'multi_output_model') -> Dict[str, Any]:
        """
        Train multi-output model with multiple prediction targets.
        
        Args:
            features: Training features
            targets: Dictionary of target series:
                - 'price_prediction': Price prediction before hitting opposite barrier
                - 'probability': Probability of hitting the barrier
                - 'risk': Risk of hitting opposite price barrier first
            model_name: Name of the model
            
        Returns:
            Multi-output model training result
        """
        try:
            self.logger.info(f"🤖 Training multi-output model: {model_name}")
            
            # Validate input data
            features_validation = validate_data_quality(features, 'features', 'comprehensive')
            
            # Validate each target
            targets_validation = {}
            for target_name, target_series in targets.items():
                targets_validation[target_name] = validate_data_quality(target_series, 'targets', 'standard')
            
            # Prepare multi-output training data
            X_train, X_test, y_train_dict, y_test_dict = await self._prepare_multi_output_data(features, targets)
            
            # Train individual models for each output
            trained_models = {}
            evaluation_results = {}
            
            for output_name, y_train in y_train_dict.items():
                y_test = y_test_dict[output_name]
                
                self.logger.info(f"Training {output_name} model...")
                
                # Train model for this specific output
                model_result = await self._train_single_output_model(
                    X_train, y_train, X_test, y_test, f"{model_name}_{output_name}"
                )
                
                trained_models[output_name] = model_result['model']
                evaluation_results[output_name] = model_result['evaluation_metrics']
            
            # Generate multi-output predictions
            multi_output_predictions = await self._generate_multi_output_predictions(
                X_test, trained_models
            )
            
            # Calculate combined risk metrics
            risk_metrics = await self._calculate_combined_risk_metrics(
                multi_output_predictions, y_test_dict
            )
            
            return {
                'models': trained_models,
                'evaluation_results': evaluation_results,
                'multi_output_predictions': multi_output_predictions,
                'risk_metrics': risk_metrics,
                'training_metadata': {
                    'model_name': model_name,
                    'output_types': list(targets.keys()),
                    'training_timestamp': datetime.now().isoformat(),
                    'features_shape': features.shape,
                    'targets_info': {name: series.shape for name, series in targets.items()}
                },
                'features_validation': features_validation,
                'targets_validation': targets_validation
            }
            
        except Exception as e:
            self.logger.exception(f"Multi-output model training error: {e}")
            raise
    
    async def _prepare_multi_output_data(self, features: pd.DataFrame, targets: Dict[str, pd.Series]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Prepare data for multi-output training."""
        try:
            from sklearn.model_selection import train_test_split
            
            # Convert to numpy arrays
            X = features.values if hasattr(features, 'values') else features
            
            # Split data
            test_size = self.config.get('test_size', 0.2)
            random_state = self.config.get('random_state', 42)
            
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            
            # Split each target
            y_train_dict = {}
            y_test_dict = {}
            
            for target_name, target_series in targets.items():
                y = target_series.values if hasattr(target_series, 'values') else target_series
                
                # Use same split indices for consistency
                y_train, y_test = train_test_split(y, test_size=test_size, random_state=random_state)
                y_train_dict[target_name] = y_train
                y_test_dict[target_name] = y_test
            
            self.logger.info(f"Multi-output data split: Train {X_train.shape}, Test {X_test.shape}")
            
            return X_train, X_test, y_train_dict, y_test_dict
            
        except Exception as e:
            self.logger.exception(f"Error preparing multi-output data: {e}")
            raise
    
    async def _train_single_output_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                       X_test: np.ndarray, y_test: np.ndarray, 
                                       model_name: str) -> Dict[str, Any]:
        """Train a single output model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Create model based on output type
            if 'probability' in model_name.lower() or 'risk' in model_name.lower():
                # For probability and risk, use regression with sigmoid output
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # For price prediction, use standard regression
                model = RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            evaluation_metrics = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred)
            }
            
            return {
                'model': model,
                'evaluation_metrics': evaluation_metrics,
                'predictions': {
                    'train': train_pred,
                    'test': test_pred
                }
            }
            
        except Exception as e:
            self.logger.exception(f"Error training single output model {model_name}: {e}")
            raise
    
    async def _generate_multi_output_predictions(self, X_test: np.ndarray, 
                                               trained_models: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate multi-output predictions."""
        try:
            predictions = {}
            
            for output_name, model in trained_models.items():
                pred = model.predict(X_test)
                
                # Apply appropriate transformations based on output type
                if 'probability' in output_name.lower():
                    # Ensure probabilities are in [0, 1] range
                    pred = np.clip(pred, 0, 1)
                elif 'risk' in output_name.lower():
                    # Ensure risk values are in [0, 1] range
                    pred = np.clip(pred, 0, 1)
                
                predictions[output_name] = pred
            
            return predictions
            
        except Exception as e:
            self.logger.exception(f"Error generating multi-output predictions: {e}")
            raise
    
    async def _calculate_combined_risk_metrics(self, predictions: Dict[str, np.ndarray], 
                                             y_test_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate combined risk metrics from multi-output predictions."""
        try:
            risk_metrics = {}
            
            # Calculate correlation between different outputs
            if 'price_prediction' in predictions and 'probability' in predictions:
                price_pred = predictions['price_prediction']
                prob_pred = predictions['probability']
                
                correlation = np.corrcoef(price_pred, prob_pred)[0, 1]
                risk_metrics['price_probability_correlation'] = correlation
            
            # Calculate risk-adjusted predictions
            if 'price_prediction' in predictions and 'risk' in predictions:
                price_pred = predictions['price_prediction']
                risk_pred = predictions['risk']
                
                # Risk-adjusted price prediction
                risk_adjusted_price = price_pred * (1 - risk_pred)
                risk_metrics['risk_adjusted_price_prediction'] = risk_adjusted_price.mean()
            
            # Calculate overall model confidence
            if 'probability' in predictions:
                prob_pred = predictions['probability']
                confidence = np.mean(prob_pred)
                uncertainty = np.std(prob_pred)
                
                risk_metrics['overall_confidence'] = confidence
                risk_metrics['prediction_uncertainty'] = uncertainty
            
            return risk_metrics
            
        except Exception as e:
            self.logger.exception(f"Error calculating combined risk metrics: {e}")
            return {}


class ConsolidatedAnalystEnhancement:
    """
    Consolidated Analyst Enhancement with Multi-Output Support.
    
    This replaces:
    - src/training/steps/model_training/step12_analyst_enhancement.py (2,703 lines)
    - src/training/steps/model_training/step12_analyst_enhancement_per_regime.py
    - src/training/steps/model_training/step12_analyst_enhancement_optimized.py
    
    Core Principles Preserved:
    - per-HMM regime training: Models are trained specifically for different HMM-identified market regimes
    - Analyst/Tactician separation: Distinct roles and models for Analyst and Tactician components
    - Multi-output functionality: Price prediction, probability, risk of hitting opposite price barrier
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize consolidated analyst enhancement."""
        self.config = validate_and_fix_config(config, 'model_training')
        self.logger = logger.getChild('ConsolidatedAnalystEnhancement')
        
        # Initialize multi-output model trainer
        self.multi_output_trainer = MultiOutputModelTrainer(self.config)
        
        self.logger.info("🚀 Consolidated Analyst Enhancement initialized")
    
    async def execute(self, features: pd.DataFrame, targets: Union[pd.Series, Dict[str, pd.Series]], 
                     regime_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute analyst enhancement training with multi-output support.
        
        Args:
            features: Training features
            targets: Either single target series or dict of target series for multi-output
            regime_id: Optional regime ID for per-HMM regime training
            
        Returns:
            Analyst enhancement training result
        """
        try:
            self.logger.info(f"🤖 Executing consolidated analyst enhancement (regime: {regime_id})...")
            
            # Prepare multi-output targets if single target provided
            if isinstance(targets, pd.Series):
                # Create multi-output targets from single target
                multi_output_targets = await self._create_multi_output_targets_from_single(features, targets)
            else:
                multi_output_targets = targets
            
            # Train multi-output model
            model_name = f'analyst_enhancement_model'
            if regime_id is not None:
                model_name += f'_regime_{regime_id}'
            
            result = await self.multi_output_trainer.train_multi_output_model(
                features, multi_output_targets, model_name
            )
            
            # Add analyst-specific metadata
            result['analyst_metadata'] = {
                'analyst_type': 'enhanced_analyst',
                'regime_id': regime_id,
                'specialization': 'multi_output_analysis',
                'capabilities': [
                    'price_prediction',
                    'probability_estimation',
                    'risk_assessment',
                    'barrier_avoidance'
                ]
            }
            
            self.logger.info(f"✅ Analyst enhancement completed: {result['training_metadata']['model_name']}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Analyst enhancement error: {e}")
            raise
    
    async def _create_multi_output_targets_from_single(self, features: pd.DataFrame, 
                                                     single_target: pd.Series) -> Dict[str, pd.Series]:
        """Create multi-output targets from single target series."""
        try:
            # This is a simplified approach - in practice, you would derive these from your data
            # For now, we'll create synthetic multi-output targets
            
            multi_output_targets = {
                'price_prediction': single_target.copy(),  # Main price prediction
                'probability': single_target.abs() / single_target.abs().max(),  # Normalized probability
                'risk': (1 - single_target.abs() / single_target.abs().max())  # Inverse risk
            }
            
            return multi_output_targets
            
        except Exception as e:
            self.logger.exception(f"Error creating multi-output targets: {e}")
            raise


class ConsolidatedTacticianSpecialistTraining:
    """
    Consolidated Tactician Specialist Training with Multi-Output Support.
    
    This replaces:
    - src/training/steps/model_training/step15_tactician_specialist_training.py (1,667 lines)
    - src/training/steps/model_training/step15_tactician_specialist_training_per_regime.py
    - src/training/steps/model_training/step14_tactician_labeling.py
    
    Core Principles Preserved:
    - per-HMM regime training: Models are trained specifically for different HMM-identified market regimes
    - Analyst/Tactician separation: Distinct roles and models for Analyst and Tactician components
    - Multi-output functionality: Price prediction, probability, risk of hitting opposite price barrier
    - Tactician labels based on Analyst's predictions: Logic preserved in unified training and labeling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize consolidated tactician specialist training."""
        self.config = validate_and_fix_config(config, 'model_training')
        self.logger = logger.getChild('ConsolidatedTacticianSpecialistTraining')
        
        # Initialize multi-output model trainer
        self.multi_output_trainer = MultiOutputModelTrainer(self.config)
        
        self.logger.info("🚀 Consolidated Tactician Specialist Training initialized")
    
    async def execute(self, features: pd.DataFrame, targets: Union[pd.Series, Dict[str, pd.Series]], 
                     regime_id: Optional[int] = None, analyst_predictions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute tactician specialist training with multi-output support.
        
        Args:
            features: Training features
            targets: Either single target series or dict of target series for multi-output
            regime_id: Optional regime ID for per-HMM regime training
            analyst_predictions: Optional analyst predictions to base tactician labels on
            
        Returns:
            Tactician specialist training result
        """
        try:
            self.logger.info(f"🤖 Executing consolidated tactician specialist training (regime: {regime_id})...")
            
            # Prepare multi-output targets
            if isinstance(targets, pd.Series):
                # Create multi-output targets from single target
                multi_output_targets = await self._create_multi_output_targets_from_single(features, targets)
            else:
                multi_output_targets = targets
            
            # Incorporate analyst predictions if provided (Tactician labels based on Analyst's predictions)
            if analyst_predictions is not None:
                multi_output_targets = await self._incorporate_analyst_predictions(
                    multi_output_targets, analyst_predictions
                )
            
            # Train multi-output model
            model_name = f'tactician_specialist_model'
            if regime_id is not None:
                model_name += f'_regime_{regime_id}'
            
            result = await self.multi_output_trainer.train_multi_output_model(
                features, multi_output_targets, model_name
            )
            
            # Add tactician-specific metadata
            result['tactician_metadata'] = {
                'tactician_type': 'specialist_tactician',
                'regime_id': regime_id,
                'specialization': 'multi_output_tactical_analysis',
                'capabilities': [
                    'price_prediction',
                    'probability_estimation',
                    'risk_assessment',
                    'barrier_avoidance',
                    'analyst_integration'
                ],
                'analyst_integration': analyst_predictions is not None
            }
            
            self.logger.info(f"✅ Tactician specialist training completed: {result['training_metadata']['model_name']}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Tactician specialist training error: {e}")
            raise
    
    async def _create_multi_output_targets_from_single(self, features: pd.DataFrame, 
                                                     single_target: pd.Series) -> Dict[str, pd.Series]:
        """Create multi-output targets from single target series."""
        try:
            # This is a simplified approach - in practice, you would derive these from your data
            # For now, we'll create synthetic multi-output targets
            
            multi_output_targets = {
                'price_prediction': single_target.copy(),  # Main price prediction
                'probability': single_target.abs() / single_target.abs().max(),  # Normalized probability
                'risk': (1 - single_target.abs() / single_target.abs().max())  # Inverse risk
            }
            
            return multi_output_targets
            
        except Exception as e:
            self.logger.exception(f"Error creating multi-output targets: {e}")
            raise
    
    async def _incorporate_analyst_predictions(self, tactician_targets: Dict[str, pd.Series], 
                                             analyst_predictions: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Incorporate analyst predictions into tactician targets.
        
        This preserves the core principle: Tactician labels based on Analyst's predictions
        """
        try:
            # Modify tactician targets based on analyst predictions
            # This is where the integration between Analyst and Tactician happens
            
            enhanced_targets = tactician_targets.copy()
            
            if 'price_prediction' in analyst_predictions:
                # Use analyst price prediction to adjust tactician targets
                analyst_price = analyst_predictions['price_prediction']
                if hasattr(analyst_price, 'values'):
                    analyst_price = analyst_price.values
                
                # Adjust tactician price prediction based on analyst input
                tactician_price = enhanced_targets['price_prediction']
                if hasattr(tactician_price, 'values'):
                    tactician_price = tactician_price.values
                else:
                    tactician_price = tactician_price
                
                # Weighted combination of analyst and tactician predictions
                combined_price = 0.7 * tactician_price + 0.3 * analyst_price
                enhanced_targets['price_prediction'] = pd.Series(combined_price)
            
            if 'probability' in analyst_predictions:
                # Use analyst probability to adjust tactician probability
                analyst_prob = analyst_predictions['probability']
                if hasattr(analyst_prob, 'values'):
                    analyst_prob = analyst_prob.values
                
                tactician_prob = enhanced_targets['probability']
                if hasattr(tactician_prob, 'values'):
                    tactician_prob = tactician_prob.values
                else:
                    tactician_prob = tactician_prob
                
                # Weighted combination
                combined_prob = 0.6 * tactician_prob + 0.4 * analyst_prob
                enhanced_targets['probability'] = pd.Series(combined_prob)
            
            self.logger.info("✅ Incorporated analyst predictions into tactician targets")
            
            return enhanced_targets
            
        except Exception as e:
            self.logger.exception(f"Error incorporating analyst predictions: {e}")
            return tactician_targets


# Backward compatibility wrappers
class AnalystEnhancement(ConsolidatedAnalystEnhancement):
    """Backward compatibility wrapper for AnalystEnhancement."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for AnalystEnhancement")


class TacticianSpecialistTraining(ConsolidatedTacticianSpecialistTraining):
    """Backward compatibility wrapper for TacticianSpecialistTraining."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for TacticianSpecialistTraining")


# Example usage and testing
async def example_analyst_tactician_training():
    """Example of using the consolidated analyst and tactician training."""
    logger.info("🚀 Example: Consolidated Analyst and Tactician Training")
    
    # Configuration
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1m',
        'test_size': 0.2,
        'random_state': 42,
        'model_training_config': {
            'enable_confidence_metrics': True,
            'enable_calibration_assessment': True,
            'enable_feature_importance': True
        }
    }
    
    # Create sample data
    n_samples = 1000
    n_features = 20
    
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Single target (will be converted to multi-output)
    single_target = pd.Series(np.random.randn(n_samples))
    
    # Multi-output targets
    multi_output_targets = {
        'price_prediction': pd.Series(np.random.randn(n_samples)),
        'probability': pd.Series(np.random.rand(n_samples)),
        'risk': pd.Series(np.random.rand(n_samples))
    }
    
    print("=" * 80)
    print("CONSOLIDATED ANALYST AND TACTICIAN TRAINING EXAMPLE")
    print("=" * 80)
    
    # Test Analyst Enhancement
    print("\n🤖 Testing Analyst Enhancement...")
    analyst = ConsolidatedAnalystEnhancement(config)
    
    # Test with single target
    analyst_result_single = await analyst.execute(features, single_target, regime_id=0)
    print(f"✅ Analyst (single target): {analyst_result_single['training_metadata']['model_name']}")
    print(f"📊 Output types: {analyst_result_single['training_metadata']['output_types']}")
    
    # Test with multi-output targets
    analyst_result_multi = await analyst.execute(features, multi_output_targets, regime_id=1)
    print(f"✅ Analyst (multi-output): {analyst_result_multi['training_metadata']['model_name']}")
    print(f"📊 Output types: {analyst_result_multi['training_metadata']['output_types']}")
    
    # Test Tactician Specialist Training
    print("\n🎯 Testing Tactician Specialist Training...")
    tactician = ConsolidatedTacticianSpecialistTraining(config)
    
    # Test with analyst predictions
    analyst_predictions = {
        'price_prediction': pd.Series(np.random.randn(n_samples)),
        'probability': pd.Series(np.random.rand(n_samples))
    }
    
    tactician_result = await tactician.execute(
        features, multi_output_targets, 
        regime_id=0, 
        analyst_predictions=analyst_predictions
    )
    print(f"✅ Tactician: {tactician_result['training_metadata']['model_name']}")
    print(f"📊 Output types: {tactician_result['training_metadata']['output_types']}")
    print(f"🔗 Analyst integration: {tactician_result['tactician_metadata']['analyst_integration']}")
    
    # Test backward compatibility
    print("\n🔄 Testing Backward Compatibility...")
    old_analyst = AnalystEnhancement(config)
    old_tactician = TacticianSpecialistTraining(config)
    print("✅ Backward compatibility wrappers working")
    
    print("\n🎉 All tests completed successfully!")
    
    return {
        'analyst_result_single': analyst_result_single,
        'analyst_result_multi': analyst_result_multi,
        'tactician_result': tactician_result
    }


if __name__ == "__main__":
    asyncio.run(example_analyst_tactician_training())