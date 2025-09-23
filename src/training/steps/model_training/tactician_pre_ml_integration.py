"""
Tactician Pre-ML Integration Component

This module integrates the Tactician Pre-ML Training Orchestrator with the existing
models_training sub_pipeline, providing a seamless interface for the dual-directional
training approach.

Key Features:
- Integration with existing sub_pipeline architecture
- Automatic signal separation and feature optimization
- Dual training execution (longs vs shorts)
- Comprehensive error handling and logging
- Backward compatibility with existing training steps
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

# Core imports
from src.utils.logger import get_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.core.decorators import handles_errors, traced, log_execution_time, validates

# Import the orchestrator
from .tactician_pre_ml_orchestrator import (
    TacticianPreMLOrchestrator, TacticianPreMLConfig, TacticianPreMLResult
)

# Import existing training components for compatibility
from .sub_pipeline import SubPipelineConfig
from .base_step import BaseTrainingStep

class TacticianPreMLIntegrationStep(BaseTrainingStep):
    """
    Integration step for Tactician Pre-ML Training Orchestrator.
    
    This class provides a bridge between the existing models_training sub_pipeline
    and the new dual-directional training approach.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Tactician Pre-ML Integration Step."""
        super().__init__(config or {})
        self.logger = get_logger('TacticianPreMLIntegrationStep')
        
        # Initialize orchestrator configuration
        self.orchestrator_config = self._create_orchestrator_config()
        self.orchestrator = TacticianPreMLOrchestrator(self.orchestrator_config)
        
        self.logger.info("🚀 TacticianPreMLIntegrationStep initialized")
        self.logger.info(f"   → Confidence threshold: {self.orchestrator_config.confidence_threshold}")
        self.logger.info(f"   → Subsequent minutes: {self.orchestrator_config.subsequent_minutes}")
    
    def _create_orchestrator_config(self) -> TacticianPreMLConfig:
        """Create orchestrator configuration from step config."""
        return TacticianPreMLConfig(
            confidence_threshold=self.config.get('confidence_threshold', 0.5),
            subsequent_minutes=self.config.get('subsequent_minutes', 45),
            
            # Feature optimization settings
            enable_lookback_optimization=self.config.get('enable_lookback_optimization', True),
            enable_pid_feature_generation=self.config.get('enable_pid_feature_generation', True),
            enable_horizon_labeling=self.config.get('enable_horizon_labeling', True),
            enable_feature_selection=self.config.get('enable_feature_selection', True),
            
            # Training settings
            enable_base_training=self.config.get('enable_base_training', True),
            enable_ensemble_training=self.config.get('enable_ensemble_training', True),
            
            # Data processing
            max_samples_per_direction=self.config.get('max_samples_per_direction', None),
            enable_data_validation=self.config.get('enable_data_validation', True),
            enable_progress_logging=self.config.get('enable_progress_logging', True),
            
            # Output settings
            save_intermediate_results=self.config.get('save_intermediate_results', True),
            output_directory=self.config.get('output_directory', "generated/tactician_pre_ml_training"),
            
            # Feature optimization configs
            lookback_config=self.config.get('lookback_config', {}),
            pid_config=self.config.get('pid_config', {}),
            horizon_config=self.config.get('horizon_config', {}),
            feature_selection_config=self.config.get('feature_selection_config', {}),
            
            # Training configs
            base_training_config=self.config.get('base_training_config', {}),
            ensemble_training_config=self.config.get('ensemble_training_config', {})
        )
    
    @traced(span_name='execute_tactician_pre_ml_integration')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=False)
    @log_execution_time()
    async def execute(self, 
                     data: Dict[str, Any],
                     analyst_outputs: Dict[str, Any],
                     **kwargs) -> Dict[str, Any]:
        """
        Execute Tactician Pre-ML training integration.
        
        Args:
            data: Market data dictionary
            analyst_outputs: Analyst outputs dictionary
            **kwargs: Additional parameters
            
        Returns:
            Dict containing training results and metadata
        """
        start_time = time.time()
        
        self.logger.info("🚀 Starting Tactician Pre-ML Integration Step")
        self.logger.info("=" * 80)
        
        try:
            # Extract data and analyst outputs
            market_data = self._extract_market_data(data)
            analyst_data = self._extract_analyst_outputs(analyst_outputs)
            
            if market_data is None or analyst_data is None:
                raise ValueError("Invalid data provided - market data and analyst outputs are required")
            
            self.logger.info(f"   → Market data shape: {market_data.shape}")
            self.logger.info(f"   → Analyst outputs shape: {analyst_data.shape}")
            
            # Execute the full orchestration
            tprint_info("🎯 Executing Tactician Pre-ML Training Orchestration")
            result = await self.orchestrator.execute_full_orchestration(market_data, analyst_data)
            
            # Process results for sub_pipeline compatibility
            processed_results = self._process_results_for_sub_pipeline(result)
            
            execution_time = time.time() - start_time
            
            # Log success
            tprint_success("✅ Tactician Pre-ML Integration Step completed successfully")
            tprint_info(f"   → Execution time: {execution_time:.2f}s")
            tprint_info(f"   → Long models: {len(processed_results.get('long_models', {}))}")
            tprint_info(f"   → Short models: {len(processed_results.get('short_models', {}))}")
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"❌ Tactician Pre-ML Integration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _extract_market_data(self, data: Dict[str, Any]) -> Optional[Any]:
        """Extract market data from the input dictionary."""
        try:
            # Try different possible keys for market data
            possible_keys = ['market_data', 'data', 'features', 'market_features', 'ohlcv_data']
            
            for key in possible_keys:
                if key in data and data[key] is not None:
                    return data[key]
            
            # If no standard key found, return the first non-None value
            for key, value in data.items():
                if value is not None and hasattr(value, 'shape'):
                    self.logger.info(f"   → Using market data from key: {key}")
                    return value
            
            self.logger.warning("⚠️ No market data found in input")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract market data: {e}")
            return None
    
    def _extract_analyst_outputs(self, analyst_outputs: Dict[str, Any]) -> Optional[Any]:
        """Extract analyst outputs from the input dictionary."""
        try:
            # Try different possible keys for analyst outputs
            possible_keys = ['analyst_outputs', 'analyst_data', 'analyst_predictions', 'analyst_results']
            
            for key in possible_keys:
                if key in analyst_outputs and analyst_outputs[key] is not None:
                    return analyst_outputs[key]
            
            # If no standard key found, return the first non-None value
            for key, value in analyst_outputs.items():
                if value is not None and hasattr(value, 'shape'):
                    self.logger.info(f"   → Using analyst outputs from key: {key}")
                    return value
            
            self.logger.warning("⚠️ No analyst outputs found in input")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract analyst outputs: {e}")
            return None
    
    def _process_results_for_sub_pipeline(self, result: TacticianPreMLResult) -> Dict[str, Any]:
        """Process orchestrator results for sub_pipeline compatibility."""
        try:
            # Create sub_pipeline compatible results
            processed_results = {
                'success': True,
                'execution_time': result.total_processing_time,
                'configuration': result.configuration,
                
                # Signal separation results
                'signal_separation': {
                    'total_samples': result.signal_separation_result.total_samples,
                    'long_samples': result.signal_separation_result.long_samples,
                    'short_samples': result.signal_separation_result.short_samples,
                    'confidence_threshold': result.signal_separation_result.confidence_threshold,
                    'separation_time': result.signal_separation_result.separation_time
                },
                
                # Long training results
                'long_training': {
                    'direction': result.long_training_result.direction,
                    'base_models': result.long_training_result.base_models,
                    'ensemble_models': result.long_training_result.ensemble_models,
                    'training_metrics': result.long_training_result.training_metrics,
                    'model_performance': result.long_training_result.model_performance,
                    'training_time': result.long_training_result.training_time
                },
                
                # Short training results
                'short_training': {
                    'direction': result.short_training_result.direction,
                    'base_models': result.short_training_result.base_models,
                    'ensemble_models': result.short_training_result.ensemble_models,
                    'training_metrics': result.short_training_result.training_metrics,
                    'model_performance': result.short_training_result.model_performance,
                    'training_time': result.short_training_result.training_time
                },
                
                # Combined models for backward compatibility
                'models': self._combine_models(result),
                'metrics': self._combine_metrics(result),
                'performance': self._combine_performance(result)
            }
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process results for sub_pipeline: {e}")
            return {
                'success': False,
                'error': f"Result processing failed: {str(e)}"
            }
    
    def _combine_models(self, result: TacticianPreMLResult) -> Dict[str, Any]:
        """Combine models from both directions for backward compatibility."""
        combined_models = {}
        
        # Add long models
        if result.long_training_result.base_models:
            combined_models.update({
                f"long_{name}": model 
                for name, model in result.long_training_result.base_models.items()
            })
        
        if result.long_training_result.ensemble_models:
            combined_models.update({
                f"long_{name}": model 
                for name, model in result.long_training_result.ensemble_models.items()
            })
        
        # Add short models
        if result.short_training_result.base_models:
            combined_models.update({
                f"short_{name}": model 
                for name, model in result.short_training_result.base_models.items()
            })
        
        if result.short_training_result.ensemble_models:
            combined_models.update({
                f"short_{name}": model 
                for name, model in result.short_training_result.ensemble_models.items()
            })
        
        return combined_models
    
    def _combine_metrics(self, result: TacticianPreMLResult) -> Dict[str, Any]:
        """Combine metrics from both directions."""
        combined_metrics = {
            'total_processing_time': result.total_processing_time,
            'signal_separation_metrics': result.signal_separation_result.get_summary(),
            'long_training_metrics': result.long_training_result.training_metrics,
            'short_training_metrics': result.short_training_result.training_metrics
        }
        
        return combined_metrics
    
    def _combine_performance(self, result: TacticianPreMLResult) -> Dict[str, Any]:
        """Combine performance metrics from both directions."""
        combined_performance = {
            'long_model_performance': result.long_training_result.model_performance,
            'short_model_performance': result.short_training_result.model_performance,
            'overall_performance': {
                'long_avg_score': self._calculate_average_score(result.long_training_result.model_performance),
                'short_avg_score': self._calculate_average_score(result.short_training_result.model_performance),
                'total_models': (
                    len(result.long_training_result.base_models) + 
                    len(result.long_training_result.ensemble_models) +
                    len(result.short_training_result.base_models) + 
                    len(result.short_training_result.ensemble_models)
                )
            }
        }
        
        return combined_performance
    
    def _calculate_average_score(self, model_performance: Dict[str, Any]) -> float:
        """Calculate average score from model performance metrics."""
        try:
            scores = []
            
            # Extract scores from nested dictionaries
            for key, value in model_performance.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)) and 'score' in sub_key.lower():
                            scores.append(sub_value)
                elif isinstance(value, (int, float)) and 'score' in key.lower():
                    scores.append(value)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception:
            return 0.0

class TacticianPreMLSubPipelineIntegration:
    """
    Integration class for adding Tactician Pre-ML training to the sub_pipeline.
    
    This class provides methods to integrate the Pre-ML orchestrator into the
    existing models_training sub_pipeline architecture.
    """
    
    def __init__(self, sub_pipeline):
        """Initialize the integration with the sub_pipeline."""
        self.sub_pipeline = sub_pipeline
        self.logger = get_logger('TacticianPreMLSubPipelineIntegration')
        
        self.logger.info("🔗 TacticianPreMLSubPipelineIntegration initialized")
    
    def add_pre_ml_training_step(self, 
                                step_name: str = "tactician_pre_ml_training",
                                config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add Tactician Pre-ML training step to the sub_pipeline.
        
        Args:
            step_name: Name for the new training step
            config: Configuration for the Pre-ML training
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            # Create the integration step
            integration_step = TacticianPreMLIntegrationStep(config)
            
            # Add to sub_pipeline
            if hasattr(self.sub_pipeline, 'add_training_step'):
                self.sub_pipeline.add_training_step(step_name, integration_step)
            else:
                # Fallback: add to available steps
                if not hasattr(self.sub_pipeline, 'available_steps'):
                    self.sub_pipeline.available_steps = {}
                
                self.sub_pipeline.available_steps[step_name] = integration_step
            
            self.logger.info(f"✅ Added Tactician Pre-ML training step: {step_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add Pre-ML training step: {e}")
            return False
    
    def create_pre_ml_sub_pipeline_config(self, 
                                        symbol: str,
                                        exchange: str,
                                        timeframe: str = "1m",
                                        config: Optional[Dict[str, Any]] = None) -> SubPipelineConfig:
        """
        Create a sub_pipeline configuration for Tactician Pre-ML training.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            config: Additional configuration
            
        Returns:
            SubPipelineConfig: Configuration for the sub_pipeline
        """
        default_config = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'mode': 'FULL',  # Always use FULL mode for Pre-ML training
            'steps': ['tactician_pre_ml_training'],
            'enable_logging': True,
            'save_results': True,
            'output_directory': f"generated/tactician_pre_ml_training/{symbol}_{timeframe}"
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        return SubPipelineConfig(**default_config)
    
    async def execute_pre_ml_training(self,
                                    symbol: str,
                                    exchange: str,
                                    timeframe: str = "1m",
                                    data: Optional[Dict[str, Any]] = None,
                                    analyst_outputs: Optional[Dict[str, Any]] = None,
                                    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Tactician Pre-ML training through the sub_pipeline.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data: Market data (optional, will be loaded if not provided)
            analyst_outputs: Analyst outputs (optional, will be loaded if not provided)
            config: Additional configuration
            
        Returns:
            Dict containing training results
        """
        try:
            # Create sub_pipeline configuration
            sub_config = self.create_pre_ml_sub_pipeline_config(
                symbol, exchange, timeframe, config
            )
            
            # Prepare input data
            input_data = {
                'market_data': data,
                'analyst_outputs': analyst_outputs,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe
            }
            
            # Execute through sub_pipeline
            self.logger.info(f"🚀 Executing Tactician Pre-ML training for {symbol} on {exchange}")
            results = await self.sub_pipeline.execute_sub_pipeline(
                sub_config, input_data
            )
            
            self.logger.info("✅ Tactician Pre-ML training completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Tactician Pre-ML training execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Convenience functions for easy integration
def create_tactician_pre_ml_integration_step(config: Optional[Dict[str, Any]] = None) -> TacticianPreMLIntegrationStep:
    """Create a Tactician Pre-ML Integration Step."""
    return TacticianPreMLIntegrationStep(config)

def create_tactician_pre_ml_sub_pipeline_integration(sub_pipeline) -> TacticianPreMLSubPipelineIntegration:
    """Create Tactician Pre-ML Sub-Pipeline Integration."""
    return TacticianPreMLSubPipelineIntegration(sub_pipeline)

async def execute_tactician_pre_ml_training_integration(
    sub_pipeline,
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data: Optional[Dict[str, Any]] = None,
    analyst_outputs: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Tactician Pre-ML training through sub_pipeline integration."""
    integration = create_tactician_pre_ml_sub_pipeline_integration(sub_pipeline)
    return await integration.execute_pre_ml_training(
        symbol, exchange, timeframe, data, analyst_outputs, config
    )

# Example usage
if __name__ == "__main__":
    # Example integration configuration
    config = {
        'confidence_threshold': 0.5,
        'subsequent_minutes': 45,
        'enable_lookback_optimization': True,
        'enable_pid_feature_generation': True,
        'enable_horizon_labeling': True,
        'enable_feature_selection': True,
        'enable_base_training': True,
        'enable_ensemble_training': True,
        'max_samples_per_direction': 10000,
        'save_intermediate_results': True
    }
    
    # Create integration step
    integration_step = create_tactician_pre_ml_integration_step(config)
    
    print("✅ Tactician Pre-ML Integration Step created successfully")
    print(f"   → Configuration: {config}")
    print("   → Ready for integration with models_training sub_pipeline")