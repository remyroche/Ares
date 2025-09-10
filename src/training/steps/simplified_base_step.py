"""
Simplified Base Step Infrastructure

This module replaces the complex BaseStep class with a simple, utility-based approach
that leverages MLPipelineOrchestrator and ML Common utilities.

Key Features:
- Simple function-based steps instead of complex classes
- Automatic configuration validation using ConfigurationValidator
- Automatic data quality validation using DataQualityUtilities
- Built-in error handling and recovery
- Comprehensive logging and monitoring
- Dependency injection using Step06UtilityContainer
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from abc import ABC, abstractmethod

# Import ML Common utilities
from src.utils.ml_common import (
    ConfigurationValidator,
    DataQualityUtilities,
    MLTrainingSafeguards,
    SmartFastFailHandler
)

# Import step06 utilities for dependency injection
from src.utils.step06_utilities import (
    Step06UtilityContainer,
    get_utility_container
)

# Import common operations for robust error handling
from src.utils.common_operations import (
    get_logger,
    safe_exception_handler,
    timed_operation
)

# Import core decorators
from src.core.decorators import handles_errors, traced

logger = get_logger(__name__)


class SimplifiedStepBase:
    """
    Simplified base class for training steps using utility-based approach.
    
    This replaces the complex BaseStep class with a simple, utility-based
    approach that leverages ML Common utilities for validation, data quality,
    and error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified step with utilities."""
        self.config = config
        self.step_name = self.__class__.__name__
        self.logger = logger.getChild(self.step_name)
        
        # Initialize ML Common utilities
        self.config_validator = ConfigurationValidator(self.logger)
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        self.fast_fail_handler = SmartFastFailHandler()
        
        # Initialize utility container for dependency injection
        self.utility_container = get_utility_container(config)
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info(f"🚀 {self.step_name} initialized with utilities")
    
    def _validate_configuration(self) -> None:
        """Validate configuration using ConfigurationValidator."""
        try:
            self.logger.info("🔍 Validating configuration...")
            
            # Use ConfigurationValidator from ml_common
            validation_result = self.config_validator.validate_ml_config(self.config)
            
            if validation_result['passed']:
                self.logger.info("✅ Configuration validation passed")
            else:
                error_msg = f"❌ Configuration validation failed: {validation_result.get('errors', [])}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            self.logger.exception(f"Configuration validation error: {e}")
            raise
    
    @abstractmethod
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step logic.
        
        Args:
            pipeline_state: Current state of the pipeline
            
        Returns:
            Step execution result
        """
        pass
    
    def validate_input_data(self, data: Any) -> Dict[str, Any]:
        """Validate input data using DataQualityUtilities."""
        try:
            if data is None:
                return {'passed': False, 'errors': ['Input data is None']}
            
            # Use DataQualityUtilities for validation
            validation_result = self.data_quality.analyze_data_quality(data)
            
            if not validation_result['passed']:
                self.logger.warning(f"⚠️ Input data quality issues: {validation_result.get('warnings', [])}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Input data validation error: {e}")
            return {'passed': False, 'errors': [str(e)]}
    
    def validate_output_data(self, data: Any) -> Dict[str, Any]:
        """Validate output data using DataQualityUtilities."""
        try:
            if data is None:
                return {'passed': False, 'errors': ['Output data is None']}
            
            # Use DataQualityUtilities for validation
            validation_result = self.data_quality.analyze_data_quality(data)
            
            if not validation_result['passed']:
                self.logger.warning(f"⚠️ Output data quality issues: {validation_result.get('warnings', [])}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Output data validation error: {e}")
            return {'passed': False, 'errors': [str(e)]}
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle errors using SmartFastFailHandler."""
        try:
            self.logger.exception(f"❌ Error in {self.step_name} {context}: {error}")
            
            # Use fast fail handler for critical errors
            if self.fast_fail_handler.should_fail_fast(error):
                self.logger.error(f"🚨 Fast fail triggered for {self.step_name}")
                raise
            
            # Return error result instead of raising
            return {
                'step_name': self.step_name,
                'status': 'failed',
                'error': str(error),
                'context': context,
                'executed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.exception(f"Error handling failed: {e}")
            raise


class SimplifiedDataProcessingStep(SimplifiedStepBase):
    """
    Simplified base class for data processing steps.
    
    Provides automatic data validation and quality checks.
    """
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing step with automatic validation."""
        try:
            self.logger.info(f"🔄 Executing data processing step: {self.step_name}")
            
            # Get data from pipeline state
            data = pipeline_state.get('data')
            if data is None:
                raise ValueError(f"No data found in pipeline state for {self.step_name}")
            
            # Validate input data
            input_validation = self.validate_input_data(data)
            
            # Process data
            processed_data = await self.process_data(data, pipeline_state)
            
            # Validate output data
            output_validation = self.validate_output_data(processed_data)
            
            return {
                'step_name': self.step_name,
                'data': processed_data,
                'input_validation': input_validation,
                'output_validation': output_validation,
                'status': 'completed',
                'executed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self.handle_error(e, "data processing")
    
    @abstractmethod
    async def process_data(self, data: Any, pipeline_state: Dict[str, Any]) -> Any:
        """
        Process the data.
        
        Args:
            data: Input data to process
            pipeline_state: Current pipeline state
            
        Returns:
            Processed data
        """
        pass


class SimplifiedModelTrainingStep(SimplifiedStepBase):
    """
    Simplified base class for model training steps.
    
    Provides automatic model training, evaluation, and validation.
    """
    
    async def execute(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training step with automatic validation."""
        try:
            self.logger.info(f"🔄 Executing model training step: {self.step_name}")
            
            # Get features and targets from pipeline state
            features = pipeline_state.get('features')
            targets = pipeline_state.get('targets')
            
            if features is None or targets is None:
                raise ValueError(f"Missing features or targets in pipeline state for {self.step_name}")
            
            # Validate input data
            features_validation = self.validate_input_data(features)
            targets_validation = self.validate_input_data(targets)
            
            # Train model
            model_result = await self.train_model(features, targets, pipeline_state)
            
            # Validate model result
            model_validation = self.validate_output_data(model_result.get('model'))
            
            return {
                'step_name': self.step_name,
                'model': model_result.get('model'),
                'evaluation_metrics': model_result.get('evaluation_metrics', {}),
                'features_validation': features_validation,
                'targets_validation': targets_validation,
                'model_validation': model_validation,
                'status': 'completed',
                'executed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self.handle_error(e, "model training")
    
    @abstractmethod
    async def train_model(self, features: Any, targets: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            features: Training features
            targets: Training targets
            pipeline_state: Current pipeline state
            
        Returns:
            Model training result
        """
        pass


# Utility functions for creating simple steps
def create_simple_step_function(step_name: str, step_logic: Callable) -> Callable:
    """
    Create a simple step function with standard signature.
    
    Args:
        step_name: Name of the step
        step_logic: The actual step logic
        
    Returns:
        Step function with standard signature
    """
    @handles_errors(fallback=False)
    @traced
    @timed_operation
    async def step_function(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple step function with error handling and utilities."""
        logger = get_logger(f"{__name__}.{step_name}")
        
        try:
            logger.info(f"🔄 Executing step: {step_name}")
            
            # Initialize utilities
            config_validator = ConfigurationValidator(logger)
            data_quality = DataQualityUtilities()
            
            # Validate configuration
            validation_result = config_validator.validate_ml_config(config)
            if not validation_result['passed']:
                raise ValueError(f"Configuration validation failed: {validation_result.get('errors', [])}")
            
            # Execute step logic
            result = await step_logic(config, pipeline_state)
            
            # Add step metadata
            result['step_name'] = step_name
            result['executed_at'] = datetime.now().isoformat()
            result['status'] = 'completed'
            
            logger.info(f"✅ Step '{step_name}' completed successfully")
            return result
            
        except Exception as e:
            logger.exception(f"❌ Error in step '{step_name}': {e}")
            return {
                'step_name': step_name,
                'status': 'failed',
                'error': str(e),
                'executed_at': datetime.now().isoformat()
            }
    
    return step_function


def create_data_processing_step_function(step_name: str, processing_logic: Callable) -> Callable:
    """
    Create a data processing step function with automatic validation.
    
    Args:
        step_name: Name of the step
        processing_logic: The data processing logic
        
    Returns:
        Data processing step function
    """
    @handles_errors(fallback=False)
    @traced
    @timed_operation
    async def data_processing_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Data processing step with automatic validation."""
        logger = get_logger(f"{__name__}.{step_name}")
        
        try:
            logger.info(f"🔄 Executing data processing step: {step_name}")
            
            # Initialize utilities
            config_validator = ConfigurationValidator(logger)
            data_quality = DataQualityUtilities()
            
            # Validate configuration
            validation_result = config_validator.validate_ml_config(config)
            if not validation_result['passed']:
                raise ValueError(f"Configuration validation failed: {validation_result.get('errors', [])}")
            
            # Get data from pipeline state
            data = pipeline_state.get('data')
            if data is None:
                raise ValueError(f"No data found in pipeline state for step '{step_name}'")
            
            # Validate input data
            input_validation = data_quality.analyze_data_quality(data)
            if not input_validation['passed']:
                logger.warning(f"⚠️ Input data quality issues: {input_validation.get('warnings', [])}")
            
            # Process data
            processed_data = await processing_logic(data, config, pipeline_state)
            
            # Validate output data
            output_validation = data_quality.analyze_data_quality(processed_data)
            if not output_validation['passed']:
                logger.warning(f"⚠️ Output data quality issues: {output_validation.get('warnings', [])}")
            
            return {
                'step_name': step_name,
                'data': processed_data,
                'input_validation': input_validation,
                'output_validation': output_validation,
                'status': 'completed',
                'executed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.exception(f"❌ Error in data processing step '{step_name}': {e}")
            return {
                'step_name': step_name,
                'status': 'failed',
                'error': str(e),
                'executed_at': datetime.now().isoformat()
            }
    
    return data_processing_step


# Example usage
if __name__ == "__main__":
    # Example of using simplified step functions
    async def example_data_collection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Example data collection logic."""
        import pandas as pd
        import numpy as np
        
        # Simulate data collection
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        return {'data': data}
    
    # Create step function
    data_collection_step = create_simple_step_function("data_collection", example_data_collection_logic)
    
    # Example usage
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1m',
        'data_dir': 'data'
    }
    
    pipeline_state = {}
    
    # Execute step
    result = asyncio.run(data_collection_step(config, pipeline_state))
    print(f"Step result: {result}")