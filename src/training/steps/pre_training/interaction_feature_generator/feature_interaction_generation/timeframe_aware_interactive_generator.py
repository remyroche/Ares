"""
Timeframe-Aware Interactive Feature Generation

This module provides timeframe-specific optimizations for interactive feature generation,
ensuring optimal performance and configuration for 5m, 15m, and 60m timeframes.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

from .interactive_feature_generation_component import (
    InteractiveFeatureGenerationComponent,
    InteractiveFeatureGenerationConfig
)


class TimeframeAwareInteractiveFeatureGenerator:
    """
    Timeframe-aware wrapper for interactive feature generation.
    
    Automatically selects optimal configuration based on timeframe:
    - 5m: Fast, high-frequency feature generation
    - 15m: Balanced feature generation for tactical trading
    - 60m: Thorough feature generation for strategic trading
    """
    
    def __init__(self):
        """Initialize the timeframe-aware generator."""
        self.generators = {}  # Cache for different timeframes
        
        tprint("🚀 Initializing Timeframe-Aware Interactive Feature Generator")
        
        # Define timeframe-specific configurations
        self.timeframe_configs = {
            '5m': {
                'feature_budget_pre': 80,      # Reduced for 5m
                'feature_budget_post': (20, 40),
                'interactions_cap': 10,        # Reduced for 5m
                'transforms_per_parent': 1,
                'lookback_ceiling_minutes': 60, # 1 hour ceiling for 5m
                'latency_budget_ms': 30,       # Tighter latency for 5m
                'max_workers': 8,              # More workers for 5m
                'batch_size': 2000,            # Larger batches for 5m
                'enable_matrix_optimization': True,
                'enable_hardware_optimization': True,
                'enable_parallel_processing': True,
                'verbose_logging': True
            },
            '15m': {
                'feature_budget_pre': 120,     # Standard for 15m
                'feature_budget_post': (30, 60),
                'interactions_cap': 15,        # Standard for 15m
                'transforms_per_parent': 1,
                'lookback_ceiling_minutes': 120, # 2 hour ceiling for 15m
                'latency_budget_ms': 50,       # Standard latency for 15m
                'max_workers': 6,              # Standard workers for 15m
                'batch_size': 1500,            # Standard batches for 15m
                'enable_matrix_optimization': True,
                'enable_hardware_optimization': True,
                'enable_parallel_processing': True,
                'verbose_logging': True
            },
            '60m': {
                'feature_budget_pre': 150,     # More features for 60m
                'feature_budget_post': (40, 80),
                'interactions_cap': 20,        # More interactions for 60m
                'transforms_per_parent': 2,    # More transforms for 60m
                'lookback_ceiling_minutes': 240, # 4 hour ceiling for 60m
                'latency_budget_ms': 100,      # Relaxed latency for 60m
                'max_workers': 4,              # Fewer workers for 60m
                'batch_size': 1000,            # Smaller batches for 60m
                'enable_matrix_optimization': True,
                'enable_hardware_optimization': True,
                'enable_parallel_processing': True,
                'verbose_logging': True
            }
        }
        
        tprint_info(f"📊 Configured for timeframes: {', '.join(self.timeframe_configs.keys())}")
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe string to standard format."""
        if not timeframe:
            return '15m'
        
        timeframe = timeframe.lower().strip()
        
        # Handle various formats
        if timeframe in ['5m', '5min', '5_min', '5_minute']:
            return '5m'
        elif timeframe in ['15m', '15min', '15_min', '15_minute']:
            return '15m'
        elif timeframe in ['60m', '60min', '60_min', '60_minute', '1h', '1hour', '1_hour']:
            return '60m'
        else:
            # Default to 15m for unknown timeframes
            tprint_warning(f"⚠️ Unknown timeframe format: {timeframe}, defaulting to 15m")
            return '15m'
    
    def _get_generator_for_timeframe(self, timeframe: str, symbol: str, exchange: str) -> InteractiveFeatureGenerationComponent:
        """Get or create generator for specific timeframe."""
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        cache_key = f"{normalized_timeframe}_{symbol}_{exchange}"
        
        if cache_key not in self.generators:
            tprint_debug(f"🔧 Creating generator for {normalized_timeframe} timeframe")
            
            # Get timeframe-specific configuration
            config_params = self.timeframe_configs[normalized_timeframe].copy()
            
            # Create configuration
            config = InteractiveFeatureGenerationConfig(
                symbol=symbol,
                exchange=exchange,
                timeframe=normalized_timeframe,
                **config_params
            )
            
            # Create component
            self.generators[cache_key] = InteractiveFeatureGenerationComponent(config)
            
            tprint_success(f"✅ Generator created for {normalized_timeframe.upper()}")
        
        return self.generators[cache_key]
    
    async def execute(self, 
                     training_input: Dict[str, Any], 
                     pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute interactive feature generation with timeframe-aware configuration.
        
        Args:
            training_input: Input data for feature generation
            pipeline_state: Pipeline state containing timeframe information
            
        Returns:
            Feature generation results
        """
        start_time = time.time()
        
        # Extract timeframe from pipeline state
        timeframe = pipeline_state.get('timeframe', '15m')
        symbol = pipeline_state.get('symbol', 'ETHUSDT')
        exchange = pipeline_state.get('exchange', 'binance')
        
        tprint(f"🚀 Starting Timeframe-Aware Interactive Feature Generation")
        tprint_info(f"   → Symbol: {symbol}")
        tprint_info(f"   → Exchange: {exchange}")
        tprint_info(f"   → Timeframe: {timeframe}")
        
        # Print timeframe-specific configuration
        self._print_timeframe_config(timeframe)
        
        try:
            # Get generator for this timeframe
            generator = self._get_generator_for_timeframe(timeframe, symbol, exchange)
            
            # Execute feature generation
            tprint_info("🔧 Executing timeframe-optimized interactive feature generation...")
            result = await generator.execute(training_input, pipeline_state)
            
            # Add timeframe information to result
            if hasattr(result, 'metadata'):
                result.metadata['timeframe'] = timeframe
                result.metadata['timeframe_optimized'] = True
                result.metadata['configuration_used'] = self.timeframe_configs[self._normalize_timeframe(timeframe)]
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Timeframe-aware feature generation completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Timeframe-aware feature generation failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error_message': error_msg,
                'execution_time': execution_time,
                'timeframe': timeframe
            }
    
    def _print_timeframe_config(self, timeframe: str):
        """Print configuration summary for a specific timeframe."""
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        if normalized_timeframe not in self.timeframe_configs:
            tprint_error(f"❌ No configuration found for timeframe: {timeframe}")
            return
        
        config = self.timeframe_configs[normalized_timeframe]
        
        tprint_info(f"📊 Configuration for {normalized_timeframe.upper()}:")
        tprint_info(f"   → Feature budget (pre): {config['feature_budget_pre']}")
        tprint_info(f"   → Feature budget (post): {config['feature_budget_post']}")
        tprint_info(f"   → Interactions cap: {config['interactions_cap']}")
        tprint_info(f"   → Transforms per parent: {config['transforms_per_parent']}")
        tprint_info(f"   → Lookback ceiling: {config['lookback_ceiling_minutes']} minutes")
        tprint_info(f"   → Latency budget: {config['latency_budget_ms']} ms")
        tprint_info(f"   → Max workers: {config['max_workers']}")
        tprint_info(f"   → Batch size: {config['batch_size']}")
        tprint_info(f"   → Matrix optimization: {config['enable_matrix_optimization']}")
        tprint_info(f"   → Hardware optimization: {config['enable_hardware_optimization']}")
    
    def get_timeframe_info(self, timeframe: str) -> Dict[str, Any]:
        """Get information about configuration for a specific timeframe."""
        normalized_timeframe = self._normalize_timeframe(timeframe)
        
        if normalized_timeframe not in self.timeframe_configs:
            return {'error': f'Timeframe {timeframe} not supported'}
        
        config = self.timeframe_configs[normalized_timeframe]
        
        return {
            'timeframe': normalized_timeframe,
            'feature_budget_pre': config['feature_budget_pre'],
            'feature_budget_post': config['feature_budget_post'],
            'interactions_cap': config['interactions_cap'],
            'transforms_per_parent': config['transforms_per_parent'],
            'lookback_ceiling_minutes': config['lookback_ceiling_minutes'],
            'latency_budget_ms': config['latency_budget_ms'],
            'max_workers': config['max_workers'],
            'batch_size': config['batch_size'],
            'enable_matrix_optimization': config['enable_matrix_optimization'],
            'enable_hardware_optimization': config['enable_hardware_optimization'],
            'enable_parallel_processing': config['enable_parallel_processing']
        }
    
    def list_supported_timeframes(self) -> List[Dict[str, Any]]:
        """List all supported timeframes with their configurations."""
        timeframes = []
        
        for timeframe in self.timeframe_configs.keys():
            info = self.get_timeframe_info(timeframe)
            timeframes.append(info)
        
        return timeframes
    
    def cleanup(self):
        """Cleanup resources."""
        tprint_debug("🧹 Cleaning up timeframe-aware generator...")
        
        for generator in self.generators.values():
            if hasattr(generator, 'cleanup'):
                generator.cleanup()
        
        self.generators.clear()
        tprint_debug("✅ Cleanup completed")


# Convenience function for direct usage
async def execute_timeframe_aware_generation(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute timeframe-aware interactive feature generation.
    
    Args:
        training_input: Input data for feature generation
        pipeline_state: Pipeline state containing timeframe information
        
    Returns:
        Feature generation results
    """
    generator = TimeframeAwareInteractiveFeatureGenerator()
    try:
        return await generator.execute(training_input, pipeline_state)
    finally:
        generator.cleanup()


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
        
        generator = TimeframeAwareInteractiveFeatureGenerator()
        
        # List supported timeframes
        tprint("📊 Supported Timeframes:")
        for info in generator.list_supported_timeframes():
            tprint_info(f"   → {info['timeframe'].upper()}: "
                       f"Budget {info['feature_budget_pre']}, "
                       f"Interactions {info['interactions_cap']}, "
                       f"Workers {info['max_workers']}")
        
        # Test each timeframe
        for pipeline_state in pipeline_states:
            tprint(f"\n🧪 Testing {pipeline_state['timeframe']} timeframe...")
            
            # Create sample training input
            training_input = {
                'data': None,  # Would be actual data in real usage
                'targets': {}
            }
            
            result = await generator.execute(training_input, pipeline_state)
            
            if hasattr(result, 'success') and result.success:
                tprint_success(f"✅ {pipeline_state['timeframe']} generation successful")
            else:
                tprint_error(f"❌ {pipeline_state['timeframe']} generation failed")
        
        generator.cleanup()
    
    asyncio.run(main())