"""Per-Regime Pipeline Integration Module.

This module provides the integration logic to ensure that steps 4-21 in the training
pipeline perform tasks on a per-HMM regime basis using consistent methods.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
import importlib
import inspect

from src.utils.logger import getChild as get_logger
from src.training.steps.regime_handler import regime_handler
from src.utils.pipeline_standards import pipeline_standards


logger = get_logger('PerRegimePipelineIntegration')


# Mapping of step names to their per-regime implementations
PER_REGIME_STEP_MAPPING = {
    'step05_labeling': 'step05_labeling_per_regime',
    'step06_feature_engineering': 'step06_feature_engineering_per_regime',
    'step07_enhanced_matrix_operations': 'step07_enhanced_matrix_operations_per_regime',
    'step08_advanced_feature_selection': 'step08_advanced_feature_selection_per_regime',
    'step09_hmm_based_training': 'step09_hmm_based_training_per_regime',
    'step10_unified_regime_intelligence': 'step10_unified_regime_intelligence_per_regime',
    'step11_analyst_creation': 'step11_analyst_creation_per_regime',
    'step12_analyst_enhancement': 'step12_analyst_enhancement_per_regime',
    'step13_analyst_ensemble_creation': 'step13_analyst_ensemble_creation_per_regime',
    'step14_tactician_labeling': 'step14_tactician_labeling_per_regime',
    'step15_tactician_specialist_training': 'step15_tactician_specialist_training_per_regime',
    'step16_confidence_calibration': 'step16_confidence_calibration_per_regime',
    'step17_final_parameters_optimization': 'step17_final_parameters_optimization_per_regime',
    'step18_walk_forward_validation': 'step18_walk_forward_validation_per_regime',
    'step19_monte_carlo_validation': 'step19_monte_carlo_validation_per_regime',
    'step20_ab_testing': 'step20_ab_testing_per_regime',
    'step21_saving': 'step21_saving_per_regime'
}


class PerRegimePipelineIntegrator:
    """Integrator for per-regime pipeline processing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pipeline integrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger('PerRegimePipelineIntegrator')
        self.per_regime_enabled = self.config.get('per_regime_processing', True)
        self.regime_handler = regime_handler
        
    def should_use_per_regime(self, step_name: str) -> bool:
        """Determine if a step should use per-regime processing.
        
        Args:
            step_name: Name of the step
            
        Returns:
            True if per-regime processing should be used
        """
        # Step 4 creates the unified regime dataset, so it doesn't need per-regime processing
        if step_name in ['step04_regime_data_splitting']:
            return False
            
        # Check if per-regime is enabled globally
        if not self.per_regime_enabled:
            return False
            
        # Check step-specific configuration
        step_config = self.config.get(step_name, {})
        return step_config.get('per_regime', True)
    
    async def get_step_function(self, step_name: str) -> Optional[Callable]:
        """Get the appropriate step function (per-regime or standard).
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step function or None if not found
        """
        try:
            if self.should_use_per_regime(step_name):
                # Check if per-regime implementation exists
                per_regime_module_name = PER_REGIME_STEP_MAPPING.get(step_name)
                
                if per_regime_module_name:
                    try:
                        # Try to import per-regime module
                        module = importlib.import_module(f'src.training.steps.{per_regime_module_name}')
                        
                        # Look for run_per_regime_step function
                        if hasattr(module, 'run_per_regime_step'):
                            self.logger.info(f"✅ Using per-regime implementation for {step_name}")
                            return module.run_per_regime_step
                        else:
                            self.logger.warning(f"⚠️ Per-regime module found but no run_per_regime_step function: {per_regime_module_name}")
                    except ImportError:
                        self.logger.info(f"📝 Per-regime implementation not yet available for {step_name}")
            
            # Fall back to standard implementation
            standard_module = importlib.import_module(f'src.training.steps.{step_name}')
            
            if hasattr(standard_module, 'run_step'):
                self.logger.info(f"📋 Using standard implementation for {step_name}")
                return standard_module.run_step
            else:
                self.logger.error(f"❌ No run_step function found in {step_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading step function for {step_name}: {e}")
            return None
    
    def update_step_config_for_regime(
        self,
        step_name: str,
        config: Dict[str, Any],
        regime_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update step configuration for regime-specific processing.
        
        Args:
            step_name: Name of the step
            config: Base configuration
            regime_id: Optional regime ID for regime-specific config
            
        Returns:
            Updated configuration
        """
        updated_config = config.copy()
        
        # Add per-regime flag
        updated_config['per_regime_processing'] = self.should_use_per_regime(step_name)
        
        # Add regime-specific parameters if available
        if regime_id is not None:
            regime_params = self.config.get('regime_specific_params', {})
            if f'regime_{regime_id}' in regime_params:
                step_regime_params = regime_params[f'regime_{regime_id}'].get(step_name, {})
                updated_config.update(step_regime_params)
                
        return updated_config
    
    async def verify_regime_data_availability(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> bool:
        """Verify that regime data is available for processing.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            True if regime data is available
        """
        try:
            # Check for unified regime dataset
            training_dir = Path(data_dir) / 'training'
            unified_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            
            if not unified_file.exists():
                self.logger.error(f"❌ Unified regime data not found: {unified_file}")
                self.logger.info("💡 Please run step04_regime_data_splitting first")
                return False
                
            # Try to load and validate
            data = await self.regime_handler.load_unified_regime_data(
                symbol, exchange, timeframe, data_dir
            )
            
            if data is None or data.empty:
                self.logger.error("❌ Regime data is empty or invalid")
                return False
                
            regime_ids = self.regime_handler.get_regime_ids(data)
            if not regime_ids:
                self.logger.error("❌ No regime IDs found in data")
                return False
                
            self.logger.info(f"✅ Regime data available with {len(regime_ids)} regimes")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verifying regime data: {e}")
            return False
    
    def generate_per_regime_config_template(self) -> Dict[str, Any]:
        """Generate a template configuration for per-regime processing.
        
        Returns:
            Configuration template
        """
        template = {
            'per_regime_processing': True,
            'regime_specific_params': {
                'regime_0': {
                    'step05_labeling': {
                        'time_barrier_minutes': 45,
                        'max_lookahead': 150,
                        'profit_take_multiplier': 0.003,
                        'stop_loss_multiplier': 0.0015
                    },
                    'step06_feature_engineering': {
                        'lookback_periods': [10, 20, 50, 100, 200],
                        'emphasis': 'trend'
                    }
                },
                'regime_1': {
                    'step05_labeling': {
                        'time_barrier_minutes': 30,
                        'max_lookahead': 100,
                        'profit_take_multiplier': 0.002,
                        'stop_loss_multiplier': 0.001
                    },
                    'step06_feature_engineering': {
                        'lookback_periods': [7, 14, 30, 60],
                        'emphasis': 'balanced'
                    }
                },
                'regime_2': {
                    'step05_labeling': {
                        'time_barrier_minutes': 20,
                        'max_lookahead': 75,
                        'profit_take_multiplier': 0.0015,
                        'stop_loss_multiplier': 0.0008
                    },
                    'step06_feature_engineering': {
                        'lookback_periods': [5, 10, 20, 30],
                        'emphasis': 'mean_reversion'
                    }
                }
            },
            'step_specific_settings': {
                'step04_regime_data_splitting': {
                    'per_regime': False  # This step creates the regime data
                },
                'step09_hmm_based_training': {
                    'per_regime': True,
                    'parallel_training': True
                },
                'step17_final_parameters_optimization': {
                    'per_regime': True,
                    'optimize_across_regimes': True
                }
            }
        }
        
        return template


# Global instance
per_regime_integrator = PerRegimePipelineIntegrator()


def create_per_regime_wrapper(original_step_func: Callable) -> Callable:
    """Create a wrapper that adds per-regime processing to an existing step.
    
    This is a utility function that can wrap any existing step function
    to add per-regime processing capabilities.
    
    Args:
        original_step_func: Original step function
        
    Returns:
        Wrapped function with per-regime processing
    """
    from src.training.steps.regime_processing_decorator import per_regime_processing
    
    # Get function signature
    sig = inspect.signature(original_step_func)
    
    # Create wrapper
    @per_regime_processing(result_type='processed_data', parallel=True)
    async def regime_processor(data: pd.DataFrame, regime_id: int, **kwargs) -> Any:
        """Process data for a single regime."""
        # Call original function with regime data
        # Note: This assumes the original function can handle partial data
        return await original_step_func(
            data=data,
            regime_id=regime_id,
            **kwargs
        )
    
    async def wrapped_function(**kwargs) -> Any:
        """Wrapped function that handles per-regime processing."""
        # Extract required parameters
        symbol = kwargs.get('symbol')
        exchange = kwargs.get('exchange')
        timeframe = kwargs.get('timeframe')
        data_dir = kwargs.get('data_dir')
        
        # Check if we should use per-regime processing
        if per_regime_integrator.should_use_per_regime(original_step_func.__name__):
            # Use regime processor
            return await regime_processor(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                **kwargs
            )
        else:
            # Use original function
            return await original_step_func(**kwargs)
    
    # Copy function metadata
    wrapped_function.__name__ = original_step_func.__name__
    wrapped_function.__doc__ = f"Per-regime wrapper for {original_step_func.__name__}\n\n{original_step_func.__doc__}"
    
    return wrapped_function


# Example usage in pipeline
async def integrate_per_regime_processing(
    pipeline_config: Dict[str, Any],
    steps_to_run: List[str]
) -> Dict[str, Callable]:
    """Integrate per-regime processing into the pipeline.
    
    Args:
        pipeline_config: Pipeline configuration
        steps_to_run: List of step names to run
        
    Returns:
        Dictionary mapping step names to their functions
    """
    integrator = PerRegimePipelineIntegrator(pipeline_config)
    
    step_functions = {}
    
    for step_name in steps_to_run:
        step_func = await integrator.get_step_function(step_name)
        
        if step_func is not None:
            step_functions[step_name] = step_func
        else:
            logger.error(f"❌ Could not load function for {step_name}")
    
    return step_functions


if __name__ == '__main__':
    async def test():
        """Test the per-regime pipeline integration."""
        # Generate example configuration
        config_template = per_regime_integrator.generate_per_regime_config_template()
        
        print("📋 Per-Regime Configuration Template:")
        import json
        print(json.dumps(config_template, indent=2))
        
        # Test regime data verification
        verified = await per_regime_integrator.verify_regime_data_availability(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        
        print(f"\n✅ Regime data verified: {verified}")
        
    asyncio.run(test())