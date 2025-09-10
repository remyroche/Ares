"""
Step Function Factories

This module provides step function factory utilities extracted from training steps
to eliminate code duplication and provide consistent step function creation across all steps.

Key Features:
- Step function creation with standard signatures
- Automatic error handling and validation
- Data processing step factories with quality validation
- Step function decorators and wrappers
- Integration with pipeline infrastructure
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

# Import pipeline infrastructure
from src.utils.ml_common.pipeline_infrastructure import (
    create_simple_step_function,
    create_data_processing_step_function
)

# Import ML Common utilities
from src.utils.ml_common import (
    DataQualityUtilities,
    MLTrainingSafeguards
)

# Import common operations
from src.utils.common_operations import get_logger

# Import core decorators
from src.core.decorators import handles_errors, traced

logger = get_logger(__name__)


class StepFunctionFactory:
    """
    Factory for creating step functions with consistent patterns.
    
    This provides common step function creation patterns and utilities
    extracted from multiple training step implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize step function factory."""
        self.config = config or {}
        self.logger = logger.getChild('StepFunctionFactory')
        
        # Initialize ML Common utilities
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        self.logger.info("🚀 Step Function Factory initialized")
    
    def create_simple_step(self, step_name: str, step_function: Callable) -> Callable:
        """
        Create a simple step function with standard signature.
        
        Args:
            step_name: Name of the step
            step_function: The actual step logic
            
        Returns:
            Wrapped step function
        """
        @traced
        async def simple_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Simple step wrapper."""
            self.logger.info(f"🔄 Executing simple step: {step_name}")
            
            try:
                result = await step_function(config, pipeline_state)
                return {
                    'step_name': step_name,
                    'result': result,
                    'status': 'completed',
                    'executed_at': datetime.now().isoformat()
                }
            except Exception as e:
                self.logger.exception(f"❌ Error in simple step '{step_name}': {e}")
                return {
                    'step_name': step_name,
                    'status': 'failed',
                    'error': str(e),
                    'executed_at': datetime.now().isoformat()
                }
        
        return simple_step
    
    def create_data_processing_step(self, step_name: str, processing_function: Callable) -> Callable:
        """
        Create a data processing step with automatic data quality validation.
        
        Args:
            step_name: Name of the step
            processing_function: The data processing logic
            
        Returns:
            Wrapped data processing step function
        """
        @traced
        async def data_processing_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Data processing step with automatic validation."""
            self.logger.info(f"🔄 Executing data processing step: {step_name}")
            
            try:
                # Get data from pipeline state
                data = pipeline_state.get('data')
                if data is None:
                    raise ValueError(f"No data found in pipeline state for step '{step_name}'")
                
                # Validate input data
                input_validation = self.data_quality.analyze_data_quality(data)
                
                if not input_validation['passed']:
                    self.logger.warning(f"⚠️ Input data quality issues: {input_validation.get('warnings', [])}")
                
                # Process data
                processed_data = await processing_function(data, config)
                
                # Validate output data
                output_validation = self.data_quality.analyze_data_quality(processed_data)
                
                if not output_validation['passed']:
                    self.logger.warning(f"⚠️ Output data quality issues: {output_validation.get('warnings', [])}")
                
                return {
                    'step_name': step_name,
                    'data': processed_data,
                    'input_validation': input_validation,
                    'output_validation': output_validation,
                    'status': 'completed',
                    'executed_at': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.exception(f"❌ Error in data processing step '{step_name}': {e}")
                return {
                    'step_name': step_name,
                    'status': 'failed',
                    'error': str(e),
                    'executed_at': datetime.now().isoformat()
                }
        
        return data_processing_step
    
    def create_validation_step(self, step_name: str, validation_function: Callable) -> Callable:
        """
        Create a validation step with automatic validation reporting.
        
        Args:
            step_name: Name of the step
            validation_function: The validation logic
            
        Returns:
            Wrapped validation step function
        """
        @traced
        async def validation_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Validation step with automatic reporting."""
            self.logger.info(f"🔄 Executing validation step: {step_name}")
            
            try:
                # Perform validation
                validation_result = await validation_function(config, pipeline_state)
                
                # Add validation metadata
                validation_result.update({
                    'step_name': step_name,
                    'validation_timestamp': datetime.now().isoformat(),
                    'status': 'completed' if validation_result.get('passed', False) else 'failed'
                })
                
                if validation_result.get('passed', False):
                    self.logger.info(f"✅ Validation passed for step '{step_name}'")
                else:
                    self.logger.warning(f"⚠️ Validation failed for step '{step_name}': {validation_result.get('errors', [])}")
                
                return validation_result
                
            except Exception as e:
                self.logger.exception(f"❌ Error in validation step '{step_name}': {e}")
                return {
                    'step_name': step_name,
                    'status': 'failed',
                    'error': str(e),
                    'passed': False,
                    'errors': [str(e)],
                    'executed_at': datetime.now().isoformat()
                }
        
        return validation_step
    
    def create_training_step(self, step_name: str, training_function: Callable) -> Callable:
        """
        Create a training step with automatic model validation and persistence.
        
        Args:
            step_name: Name of the step
            training_function: The training logic
            
        Returns:
            Wrapped training step function
        """
        @traced
        async def training_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Training step with automatic model validation."""
            self.logger.info(f"🔄 Executing training step: {step_name}")
            
            try:
                # Perform training
                training_result = await training_function(config, pipeline_state)
                
                # Validate model if present
                if 'model' in training_result:
                    model = training_result['model']
                    model_validation = self._validate_model(model)
                    training_result['model_validation'] = model_validation
                
                # Add training metadata
                training_result.update({
                    'step_name': step_name,
                    'training_timestamp': datetime.now().isoformat(),
                    'status': 'completed'
                })
                
                self.logger.info(f"✅ Training completed for step '{step_name}'")
                
                return training_result
                
            except Exception as e:
                self.logger.exception(f"❌ Error in training step '{step_name}': {e}")
                return {
                    'step_name': step_name,
                    'status': 'failed',
                    'error': str(e),
                    'executed_at': datetime.now().isoformat()
                }
        
        return training_step
    
    def create_evaluation_step(self, step_name: str, evaluation_function: Callable) -> Callable:
        """
        Create an evaluation step with automatic metric calculation and reporting.
        
        Args:
            step_name: Name of the step
            evaluation_function: The evaluation logic
            
        Returns:
            Wrapped evaluation step function
        """
        @traced
        async def evaluation_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluation step with automatic metric calculation."""
            self.logger.info(f"🔄 Executing evaluation step: {step_name}")
            
            try:
                # Perform evaluation
                evaluation_result = await evaluation_function(config, pipeline_state)
                
                # Add evaluation metadata
                evaluation_result.update({
                    'step_name': step_name,
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'status': 'completed'
                })
                
                # Log key metrics
                if 'evaluation_metrics' in evaluation_result:
                    metrics = evaluation_result['evaluation_metrics']
                    self.logger.info(f"📊 Evaluation metrics for '{step_name}': {list(metrics.keys())}")
                
                self.logger.info(f"✅ Evaluation completed for step '{step_name}'")
                
                return evaluation_result
                
            except Exception as e:
                self.logger.exception(f"❌ Error in evaluation step '{step_name}': {e}")
                return {
                    'step_name': step_name,
                    'status': 'failed',
                    'error': str(e),
                    'executed_at': datetime.now().isoformat()
                }
        
        return evaluation_step
    
    def _validate_model(self, model: Any) -> Dict[str, Any]:
        """Validate a trained model."""
        try:
            validation_result = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'model_info': {}
            }
            
            # Check if model has required methods
            required_methods = ['predict', 'fit']
            for method in required_methods:
                if not hasattr(model, method):
                    validation_result['errors'].append(f"Model missing required method: {method}")
                    validation_result['passed'] = False
            
            # Get model information
            if hasattr(model, 'get_params'):
                validation_result['model_info']['params'] = model.get_params()
            
            if hasattr(model, '__class__'):
                validation_result['model_info']['type'] = model.__class__.__name__
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"Error validating model: {e}")
            return {
                'passed': False,
                'errors': [f"Model validation error: {e}"],
                'warnings': [],
                'model_info': {}
            }
    
    def create_optimization_step(self, step_name: str, optimization_function: Callable) -> Callable:
        """
        Create an optimization step with automatic performance monitoring.
        
        Args:
            step_name: Name of the step
            optimization_function: The optimization logic
            
        Returns:
            Wrapped optimization step function
        """
        @traced
        async def optimization_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Optimization step with automatic performance monitoring."""
            self.logger.info(f"🔄 Executing optimization step: {step_name}")
            
            try:
                # Monitor performance before optimization
                import psutil
                import time
                
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Perform optimization
                optimization_result = await optimization_function(config, pipeline_state)
                
                # Monitor performance after optimization
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Add performance metrics
                optimization_result.update({
                    'step_name': step_name,
                    'optimization_timestamp': datetime.now().isoformat(),
                    'status': 'completed',
                    'performance_metrics': {
                        'execution_time_seconds': end_time - start_time,
                        'memory_usage_mb': end_memory - start_memory,
                        'peak_memory_mb': end_memory
                    }
                })
                
                self.logger.info(f"✅ Optimization completed for step '{step_name}' in {end_time - start_time:.2f}s")
                
                return optimization_result
                
            except Exception as e:
                self.logger.exception(f"❌ Error in optimization step '{step_name}': {e}")
                return {
                    'step_name': step_name,
                    'status': 'failed',
                    'error': str(e),
                    'executed_at': datetime.now().isoformat()
                }
        
        return optimization_step


# Global instance for easy access
_global_step_factory = None

def get_step_factory(config: Optional[Dict[str, Any]] = None) -> StepFunctionFactory:
    """Get step function factory instance."""
    global _global_step_factory
    if _global_step_factory is None:
        _global_step_factory = StepFunctionFactory(config)
    return _global_step_factory


# Convenience functions
def create_simple_step(step_name: str, step_function: Callable) -> Callable:
    """Create a simple step function."""
    factory = get_step_factory()
    return factory.create_simple_step(step_name, step_function)


def create_data_processing_step(step_name: str, processing_function: Callable) -> Callable:
    """Create a data processing step function."""
    factory = get_step_factory()
    return factory.create_data_processing_step(step_name, processing_function)


def create_validation_step(step_name: str, validation_function: Callable) -> Callable:
    """Create a validation step function."""
    factory = get_step_factory()
    return factory.create_validation_step(step_name, validation_function)


def create_training_step(step_name: str, training_function: Callable) -> Callable:
    """Create a training step function."""
    factory = get_step_factory()
    return factory.create_training_step(step_name, training_function)


def create_evaluation_step(step_name: str, evaluation_function: Callable) -> Callable:
    """Create an evaluation step function."""
    factory = get_step_factory()
    return factory.create_evaluation_step(step_name, evaluation_function)


def create_optimization_step(step_name: str, optimization_function: Callable) -> Callable:
    """Create an optimization step function."""
    factory = get_step_factory()
    return factory.create_optimization_step(step_name, optimization_function)


# Example usage
if __name__ == "__main__":
    import asyncio
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Test step function factory
    factory = StepFunctionFactory()
    
    # Define step functions
    async def data_collection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Data collection logic."""
        return {'data': data}
    
    async def feature_engineering_logic(data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Feature engineering logic."""
        features = data.copy()
        features['returns'] = features['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        return features
    
    async def model_training_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Model training logic."""
        from sklearn.ensemble import RandomForestClassifier
        
        features = pipeline_state['features']
        targets = pd.Series(np.random.randint(0, 2, len(features)), name='target')
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, targets)
        
        return {'model': model, 'targets': targets}
    
    async def model_evaluation_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Model evaluation logic."""
        model = pipeline_state['model']
        features = pipeline_state['features']
        targets = pipeline_state['targets']
        
        predictions = model.predict(features)
        accuracy = (predictions == targets).mean()
        
        return {'evaluation_metrics': {'accuracy': accuracy}, 'predictions': predictions}
    
    # Create step functions
    data_collection_step = factory.create_simple_step("data_collection", data_collection_logic)
    feature_engineering_step = factory.create_data_processing_step("feature_engineering", feature_engineering_logic)
    model_training_step = factory.create_training_step("model_training", model_training_logic)
    model_evaluation_step = factory.create_evaluation_step("model_evaluation", model_evaluation_logic)
    
    # Test step execution
    async def test_steps():
        config = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '1m'}
        pipeline_state = {}
        
        print("=== Testing Step Functions ===")
        
        # Execute data collection
        result1 = await data_collection_step(config, pipeline_state)
        print(f"Data collection: {result1['status']}")
        pipeline_state.update(result1)
        
        # Execute feature engineering
        result2 = await feature_engineering_step(config, pipeline_state)
        print(f"Feature engineering: {result2['status']}")
        pipeline_state.update(result2)
        
        # Execute model training
        result3 = await model_training_step(config, pipeline_state)
        print(f"Model training: {result3['status']}")
        pipeline_state.update(result3)
        
        # Execute model evaluation
        result4 = await model_evaluation_step(config, pipeline_state)
        print(f"Model evaluation: {result4['status']}")
        print(f"Accuracy: {result4['evaluation_metrics']['accuracy']:.3f}")
        
        return pipeline_state
    
    # Run test
    asyncio.run(test_steps())