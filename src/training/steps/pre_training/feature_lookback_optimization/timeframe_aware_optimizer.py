"""
Timeframe-Aware Feature Lookback Optimizer

This module provides a wrapper around the main feature lookback optimization
component that automatically selects the appropriate configuration based on
the timeframe (5m, 15m, 60m).
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

from .timeframe_config_loader import (
    get_timeframe_config_loader, 
    get_optimized_parameters, 
    validate_timeframe
)
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent


class TimeframeAwareFeatureLookbackOptimizer:
    """
    Timeframe-aware wrapper for feature lookback optimization.
    
    Automatically selects optimal configuration based on timeframe:
    - 5m: Fast, high-frequency optimization
    - 15m: Balanced optimization for tactical trading
    - 60m: Thorough optimization for strategic trading
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the timeframe-aware optimizer."""
        self.config_loader = get_timeframe_config_loader()
        self.optimizers = {}  # Cache for different timeframes
        self.config_dir = config_dir
        
        tprint("🚀 Initializing Timeframe-Aware Feature Lookback Optimizer")
        tprint_info(f"📁 Configuration directory: {self.config_dir or 'default'}")
        
        # Validate all available timeframes
        available_timeframes = self.config_loader.get_available_timeframes()
        tprint_info(f"📊 Available timeframes: {', '.join(available_timeframes)}")
        
        for timeframe in available_timeframes:
            if validate_timeframe(timeframe):
                tprint_success(f"✅ {timeframe.upper()} configuration validated")
            else:
                tprint_error(f"❌ {timeframe.upper()} configuration validation failed")
    
    def _get_optimizer_for_timeframe(self, timeframe: str) -> FeatureLookbackOptimizationComponent:
        """Get or create optimizer for specific timeframe."""
        normalized_timeframe = self.config_loader._normalize_timeframe(timeframe)
        
        if normalized_timeframe not in self.optimizers:
            tprint_debug(f"🔧 Creating optimizer for {normalized_timeframe} timeframe")
            
            # Get optimized parameters for this timeframe
            params = get_optimized_parameters(normalized_timeframe)
            
            # Create component with timeframe-specific configuration
            config = self._create_component_config(normalized_timeframe, params)
            self.optimizers[normalized_timeframe] = FeatureLookbackOptimizationComponent(config)
            
            tprint_success(f"✅ Optimizer created for {normalized_timeframe.upper()}")
        
        return self.optimizers[normalized_timeframe]
    
    def _create_component_config(self, timeframe: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create component configuration from timeframe parameters."""
        config = {
            'symbol': 'ETHUSDT',  # Default, will be overridden by pipeline state
            'exchange': 'binance',  # Default, will be overridden by pipeline state
            'timeframe': timeframe,
            'data_dir': 'historical_data',
            'custom_params': {
                'default_timeframe': timeframe,
                'base_period_minutes': params.get('base_period_minutes', 15.0),
                'min_lookback': params.get('min_lookback', 5),
                'max_lookback': params.get('max_lookback', 50),
                'lookback_step': params.get('lookback_step', 1),
                'cv_folds': params.get('cv_folds', 5),
                'max_optimization_time': params.get('max_optimization_time', 300),
                'min_samples_for_ic': params.get('min_samples_for_ic', 100),
                'min_ic_threshold': params.get('min_ic_threshold', 0.01),
                'max_workers': params.get('max_workers', 4),
                'chunk_size': params.get('chunk_size', 1000),
                'label_definition_type': params.get('label_definition_type', 'tactician'),
                'multi_target_scheme': params.get('multi_target_scheme', {}),
                'enable_matrix_optimization': True,
                'enable_hardware_optimization': True,
                'enable_parallel_processing': True,
                'verbose_logging': True
            }
        }
        
        return config
    
    async def execute(self, 
                     training_input: Optional[Dict[str, Any]], 
                     pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature lookback optimization with timeframe-aware configuration.
        
        Args:
            training_input: Input data (optional)
            pipeline_state: Pipeline state containing timeframe information
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        # Extract timeframe from pipeline state
        timeframe = pipeline_state.get('timeframe', '15m')
        symbol = pipeline_state.get('symbol', 'ETHUSDT')
        exchange = pipeline_state.get('exchange', 'binance')
        
        tprint(f"🚀 Starting Timeframe-Aware Feature Lookback Optimization")
        tprint_info(f"   → Symbol: {symbol}")
        tprint_info(f"   → Exchange: {exchange}")
        tprint_info(f"   → Timeframe: {timeframe}")
        
        # Validate timeframe
        if not validate_timeframe(timeframe):
            error_msg = f"Invalid or unsupported timeframe: {timeframe}"
            tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error_message': error_msg,
                'execution_time': time.time() - start_time
            }
        
        # Print timeframe-specific configuration summary
        self.config_loader.print_timeframe_summary(timeframe)
        
        try:
            # Get optimizer for this timeframe
            optimizer = self._get_optimizer_for_timeframe(timeframe)
            
            # Update optimizer config with pipeline state
            if hasattr(optimizer, 'config'):
                optimizer.config.symbol = symbol
                optimizer.config.exchange = exchange
                optimizer.config.timeframe = timeframe
            
            # Execute optimization
            tprint_info("🔧 Executing timeframe-optimized feature lookback optimization...")
            result = await optimizer.execute(training_input, pipeline_state)
            
            # Add timeframe information to result
            if isinstance(result, dict):
                result['timeframe'] = timeframe
                result['timeframe_optimized'] = True
                result['configuration_used'] = self.config_loader.get_optimized_parameters(timeframe)
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Timeframe-aware optimization completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Timeframe-aware optimization failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error_message': error_msg,
                'execution_time': execution_time,
                'timeframe': timeframe
            }
    
    def get_timeframe_info(self, timeframe: str) -> Dict[str, Any]:
        """Get information about configuration for a specific timeframe."""
        normalized_timeframe = self.config_loader._normalize_timeframe(timeframe)
        
        if normalized_timeframe not in self.config_loader.timeframe_configs:
            return {'error': f'Timeframe {timeframe} not supported'}
        
        config = self.config_loader.timeframe_configs[normalized_timeframe]['config']
        
        return {
            'timeframe': config.timeframe,
            'base_period_minutes': config.base_period_minutes,
            'lookback_range': f"{config.min_lookback}-{config.max_lookback} periods",
            'lookback_step': config.lookback_step,
            'cv_folds': config.cv_folds,
            'max_optimization_time': f"{config.max_optimization_time}s",
            'min_samples_for_ic': config.min_samples_for_ic,
            'min_ic_threshold': config.min_ic_threshold,
            'max_workers': config.max_workers,
            'chunk_size': config.chunk_size,
            'label_definition_type': config.label_definition_type,
            'multi_target_scheme': config.multi_target_scheme,
            'is_valid': validate_timeframe(timeframe)
        }
    
    def list_supported_timeframes(self) -> List[Dict[str, Any]]:
        """List all supported timeframes with their configurations."""
        timeframes = []
        
        for timeframe in self.config_loader.get_available_timeframes():
            info = self.get_timeframe_info(timeframe)
            timeframes.append(info)
        
        return timeframes
    
    def cleanup(self):
        """Cleanup resources."""
        tprint_debug("🧹 Cleaning up timeframe-aware optimizer...")
        
        for optimizer in self.optimizers.values():
            if hasattr(optimizer, 'cleanup'):
                optimizer.cleanup()
        
        self.optimizers.clear()
        tprint_debug("✅ Cleanup completed")


# Convenience function for direct usage
async def execute_timeframe_aware_optimization(
    training_input: Optional[Dict[str, Any]],
    pipeline_state: Dict[str, Any],
    config_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute timeframe-aware feature lookback optimization.
    
    Args:
        training_input: Input data (optional)
        pipeline_state: Pipeline state containing timeframe information
        config_dir: Optional configuration directory
        
    Returns:
        Optimization results
    """
    optimizer = TimeframeAwareFeatureLookbackOptimizer(config_dir)
    try:
        return await optimizer.execute(training_input, pipeline_state)
    finally:
        optimizer.cleanup()


# Example usage
if __name__ == "__main__":
    async def main():
        # Example pipeline states for different timeframes
        pipeline_states = [
            {
                'symbol': 'ETHUSDT',
                'exchange': 'binance',
                'timeframe': '5m',
                'data_dir': 'historical_data'
            },
            {
                'symbol': 'ETHUSDT',
                'exchange': 'binance',
                'timeframe': '15m',
                'data_dir': 'historical_data'
            },
            {
                'symbol': 'ETHUSDT',
                'exchange': 'binance',
                'timeframe': '60m',
                'data_dir': 'historical_data'
            }
        ]
        
        optimizer = TimeframeAwareFeatureLookbackOptimizer()
        
        # List supported timeframes
        tprint("📊 Supported Timeframes:")
        for info in optimizer.list_supported_timeframes():
            tprint_info(f"   → {info['timeframe'].upper()}: {info['lookback_range']} "
                       f"({info['base_period_minutes']}min periods)")
        
        # Test each timeframe
        for pipeline_state in pipeline_states:
            tprint(f"\n🧪 Testing {pipeline_state['timeframe']} timeframe...")
            result = await optimizer.execute(None, pipeline_state)
            
            if result.get('success'):
                tprint_success(f"✅ {pipeline_state['timeframe']} optimization successful")
            else:
                tprint_error(f"❌ {pipeline_state['timeframe']} optimization failed: {result.get('error_message')}")
        
        optimizer.cleanup()
    
    asyncio.run(main())