"""
Pipeline Infrastructure Utilities

This module provides core pipeline infrastructure utilities extracted from training steps
to eliminate code duplication and provide consistent pipeline management across all steps.

Key Features:
- SimplifiedPipelineManager for unified pipeline orchestration
- Step function wrapping with error handling and validation
- Pipeline execution coordination with MLPipelineOrchestrator
- Comprehensive error handling and recovery mechanisms
- Integration with ML Common utilities
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from pathlib import Path

# Import ML Common utilities
from src.utils.ml_common import (
    MLPipelineOrchestrator,
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


class SimplifiedPipelineManager:
    """
    Simplified Pipeline Manager using MLPipelineOrchestrator and utilities.
    
    This replaces the complex BaseStep approach with a simple, utility-based
    pipeline management system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simplified pipeline manager."""
        self.config = config
        self.logger = logger.getChild('SimplifiedPipelineManager')
        
        # Initialize ML Common utilities
        self.config_validator = ConfigurationValidator(self.logger)
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        self.fast_fail_handler = SmartFastFailHandler()
        
        # Initialize pipeline orchestrator
        self.orchestrator = MLPipelineOrchestrator()
        
        # Initialize utility container for dependency injection
        self.utility_container = get_utility_container(config)
        
        # Pipeline state
        self.pipeline_state = {}
        self.step_results = {}
        
        self.logger.info("🚀 Simplified Pipeline Manager initialized")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration using ML Common utilities."""
        try:
            self.logger.info("🔍 Validating configuration...")
            
            # Use ConfigurationValidator from ml_common
            validation_result = self.config_validator.validate_ml_config(self.config)
            
            if validation_result['passed']:
                self.logger.info("✅ Configuration validation passed")
                return validation_result
            else:
                error_msg = f"❌ Configuration validation failed: {validation_result.get('errors', [])}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
        except Exception as e:
            self.logger.exception(f"Configuration validation error: {e}")
            raise
    
    def add_step(self, step_name: str, step_function: Callable, 
                 dependencies: Optional[List[str]] = None,
                 timeout_seconds: Optional[int] = None) -> None:
        """
        Add a step to the pipeline.
        
        Args:
            step_name: Name of the step
            step_function: Function to execute for this step
            dependencies: List of step names this step depends on
            timeout_seconds: Timeout for step execution
        """
        try:
            # Wrap step function with error handling and utilities
            wrapped_function = self._wrap_step_function(step_function, step_name)
            
            # Add to orchestrator
            self.orchestrator.add_step(
                name=step_name,
                function=wrapped_function,
                dependencies=dependencies,
                timeout_seconds=timeout_seconds
            )
            
            self.logger.info(f"✅ Added step '{step_name}' to pipeline")
            
        except Exception as e:
            self.logger.exception(f"Error adding step '{step_name}': {e}")
            raise
    
    def _wrap_step_function(self, step_function: Callable, step_name: str) -> Callable:
        """Wrap step function with error handling, validation, and utilities."""
        
        @handles_errors(fallback=False)
        @traced
        @timed_operation
        async def wrapped_step(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Wrapped step function with comprehensive error handling and utilities."""
            try:
                self.logger.info(f"🔄 Executing step: {step_name}")
                
                # Validate input data using DataQualityUtilities
                if 'data' in pipeline_state:
                    data_validation = self.data_quality.analyze_data_quality(pipeline_state['data'])
                    if not data_validation['passed']:
                        self.logger.warning(f"⚠️ Data quality issues in {step_name}: {data_validation.get('warnings', [])}")
                
                # Execute the step function
                result = await step_function(config, pipeline_state)
                
                # Validate output data
                if isinstance(result, dict) and 'data' in result:
                    output_validation = self.data_quality.analyze_data_quality(result['data'])
                    if not output_validation['passed']:
                        self.logger.warning(f"⚠️ Output data quality issues in {step_name}: {output_validation.get('warnings', [])}")
                
                # Add step metadata
                result['step_name'] = step_name
                result['executed_at'] = datetime.now().isoformat()
                result['status'] = 'completed'
                
                self.logger.info(f"✅ Step '{step_name}' completed successfully")
                return result
                
            except Exception as e:
                self.logger.exception(f"❌ Error in step '{step_name}': {e}")
                
                # Use fast fail handler for critical errors
                if self.fast_fail_handler.should_fail_fast(e):
                    self.logger.error(f"🚨 Fast fail triggered for step '{step_name}'")
                    raise
                
                # Return error result instead of raising
                return {
                    'step_name': step_name,
                    'status': 'failed',
                    'error': str(e),
                    'executed_at': datetime.now().isoformat()
                }
        
        return wrapped_step
    
    async def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the entire pipeline using MLPipelineOrchestrator."""
        try:
            self.logger.info("🚀 Starting pipeline execution...")
            
            # Validate configuration first
            self.validate_configuration()
            
            # Execute pipeline using orchestrator
            execution_result = await self.orchestrator.execute_pipeline(
                config=self.config,
                pipeline_state=self.pipeline_state
            )
            
            # Store results
            self.step_results = execution_result.get('results', {})
            self.pipeline_state.update(self.step_results)
            
            self.logger.info("✅ Pipeline execution completed")
            return execution_result
            
        except Exception as e:
            self.logger.exception(f"❌ Pipeline execution failed: {e}")
            raise
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            'config': self.config,
            'step_results': self.step_results,
            'pipeline_state': self.pipeline_state,
            'orchestrator_status': self.orchestrator.get_status(),
            'timestamp': datetime.now().isoformat()
        }


# Utility functions for step creation
def create_simple_step_function(step_name: str, step_function: Callable) -> Callable:
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
        logger.info(f"🔄 Executing simple step: {step_name}")
        
        try:
            result = await step_function(config, pipeline_state)
            return {
                'step_name': step_name,
                'result': result,
                'status': 'completed',
                'executed_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.exception(f"❌ Error in simple step '{step_name}': {e}")
            return {
                'step_name': step_name,
                'status': 'failed',
                'error': str(e),
                'executed_at': datetime.now().isoformat()
            }
    
    return simple_step


def create_data_processing_step_function(step_name: str, processing_function: Callable) -> Callable:
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
        logger.info(f"🔄 Executing data processing step: {step_name}")
        
        try:
            # Get data from pipeline state
            data = pipeline_state.get('data')
            if data is None:
                raise ValueError(f"No data found in pipeline state for step '{step_name}'")
            
            # Validate input data
            data_quality = DataQualityUtilities()
            input_validation = data_quality.analyze_data_quality(data)
            
            if not input_validation['passed']:
                logger.warning(f"⚠️ Input data quality issues: {input_validation.get('warnings', [])}")
            
            # Process data
            processed_data = await processing_function(data, config)
            
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


# Global instance for easy access
_global_pipeline_manager = None

def get_pipeline_manager(config: Dict[str, Any]) -> SimplifiedPipelineManager:
    """Get a pipeline manager instance."""
    return SimplifiedPipelineManager(config)


# Example usage and testing
async def example_pipeline():
    """Example of how to use the simplified pipeline infrastructure."""
    
    # Configuration
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1m',
        'data_dir': 'data',
        'enable_gpu': True,
        'enable_parallel': True
    }
    
    # Create pipeline manager
    pipeline_manager = SimplifiedPipelineManager(config)
    
    # Define simple step functions
    async def step1_data_collection(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified data collection step."""
        logger.info("📊 Collecting data...")
        # Simulate data collection
        import pandas as pd
        import numpy as np
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'open': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 105,
            'low': np.random.randn(1000).cumsum() + 95,
            'close': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        return {'data': data}
    
    async def step2_feature_engineering(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified feature engineering step."""
        logger.info("🔧 Engineering features...")
        
        data = pipeline_state['data']
        
        # Simple feature engineering
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        data['sma_20'] = data['close'].rolling(20).mean()
        
        return {'features': data}
    
    # Add steps to pipeline
    pipeline_manager.add_step("data_collection", step1_data_collection)
    pipeline_manager.add_step("feature_engineering", step2_feature_engineering, 
                             dependencies=["data_collection"])
    
    # Execute pipeline
    result = await pipeline_manager.execute_pipeline()
    
    logger.info(f"Pipeline execution result: {result}")
    return result


if __name__ == "__main__":
    # Run example pipeline
    asyncio.run(example_pipeline())