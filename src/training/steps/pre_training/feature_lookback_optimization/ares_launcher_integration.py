"""
Ares Launcher Integration for Feature Lookback Optimization

This module ensures that the feature lookback optimization component
properly respects the ares_launcher configuration, including the
20-day lookback period in "light" mode.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from pathlib import Path

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)
from src.utils.data.ares_launcher_data_loader import (
    AresLauncherDataLoader,
    load_data_with_ares_mode,
    load_data_async_with_ares_mode,
    get_ares_mode_dates
)
from src.config.pipeline_modes import get_light_mode_config, get_blank_mode_config, get_full_mode_config


class AresLauncherFeatureLookbackOptimizer:
    """
    Feature lookback optimizer that respects ares_launcher configuration.
    
    This class ensures that the 20-day lookback period from "light" mode
    is properly applied when loading data for feature lookback optimization.
    """
    
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the ares launcher feature lookback optimizer."""
        self.data_loader = AresLauncherDataLoader(data_dir)
        self.data_dir = data_dir
        
        tprint("🚀 Initializing Ares Launcher Feature Lookback Optimizer")
        tprint_info(f"📁 Data directory: {data_dir}")
    
    def detect_execution_mode(self, pipeline_state: Dict[str, Any]) -> str:
        """
        Detect the execution mode from pipeline state.
        
        Args:
            pipeline_state: Pipeline state dictionary
            
        Returns:
            Detected mode ("light", "blank", "full")
        """
        # Check for explicit mode in pipeline state
        mode = pipeline_state.get('execution_mode', pipeline_state.get('mode', 'light'))
        
        # Check for lookback_days to infer mode
        lookback_days = pipeline_state.get('lookback_days')
        if lookback_days is not None:
            if lookback_days <= 30:  # Light mode typically uses 20 days
                mode = 'light'
            elif lookback_days <= 200:  # Blank mode typically uses 180 days
                mode = 'blank'
            else:  # Full mode uses 1460 days
                mode = 'full'
        
        # Check for intensity percentage
        intensity = pipeline_state.get('intensity_percentage')
        if intensity is not None:
            if intensity <= 0.05:  # Light mode
                mode = 'light'
            elif intensity <= 0.15:  # Blank mode
                mode = 'blank'
            else:  # Full mode
                mode = 'full'
        
        tprint_info(f"🔍 Detected execution mode: {mode.upper()}")
        return mode
    
    def load_data_for_optimization(
        self,
        symbol: str,
        timeframe: str,
        pipeline_state: Dict[str, Any],
        custom_start_date: Optional[datetime] = None,
        custom_end_date: Optional[datetime] = None
    ) -> Optional[Any]:
        """
        Load data for feature lookback optimization respecting ares_launcher mode.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            pipeline_state: Pipeline state dictionary
            custom_start_date: Override start date
            custom_end_date: Override end date
            
        Returns:
            Loaded DataFrame or None
        """
        # Detect execution mode
        mode = self.detect_execution_mode(pipeline_state)
        
        tprint(f"📊 Loading data for feature lookback optimization")
        tprint_info(f"   → Symbol: {symbol}")
        tprint_info(f"   → Timeframe: {timeframe}")
        tprint_info(f"   → Mode: {mode.upper()}")
        
        # Load data using ares launcher data loader
        data = self.data_loader.load_data_with_mode(
            symbol=symbol,
            interval=timeframe,
            mode=mode,
            data_type="raw",
            custom_start_date=custom_start_date,
            custom_end_date=custom_end_date
        )
        
        if data is not None and not data.empty:
            tprint_success(f"✅ Data loaded successfully for optimization")
            tprint_info(f"   → Records: {len(data)}")
            tprint_info(f"   → Date range: {data.index.min().date()} to {data.index.max().date()}")
            
            # Add mode information to data
            data.attrs['ares_mode'] = mode
            data.attrs['lookback_days'] = self._get_mode_lookback_days(mode)
            
            return data
        else:
            tprint_error(f"❌ Failed to load data for optimization")
            return None
    
    async def load_data_async_for_optimization(
        self,
        symbol: str,
        timeframe: str,
        pipeline_state: Dict[str, Any],
        custom_start_date: Optional[datetime] = None,
        custom_end_date: Optional[datetime] = None
    ) -> Optional[Any]:
        """
        Asynchronously load data for feature lookback optimization.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            pipeline_state: Pipeline state dictionary
            custom_start_date: Override start date
            custom_end_date: Override end date
            
        Returns:
            Loaded DataFrame or None
        """
        # Detect execution mode
        mode = self.detect_execution_mode(pipeline_state)
        
        tprint(f"📊 Loading data asynchronously for feature lookback optimization")
        tprint_info(f"   → Symbol: {symbol}")
        tprint_info(f"   → Timeframe: {timeframe}")
        tprint_info(f"   → Mode: {mode.upper()}")
        
        # Load data using ares launcher data loader
        data = await self.data_loader.load_data_async(
            symbol=symbol,
            interval=timeframe,
            mode=mode,
            data_type="raw",
            custom_start_date=custom_start_date,
            custom_end_date=custom_end_date
        )
        
        if data is not None and not data.empty:
            tprint_success(f"✅ Data loaded successfully for optimization")
            tprint_info(f"   → Records: {len(data)}")
            tprint_info(f"   → Date range: {data.index.min().date()} to {data.index.max().date()}")
            
            # Add mode information to data
            data.attrs['ares_mode'] = mode
            data.attrs['lookback_days'] = self._get_mode_lookback_days(mode)
            
            return data
        else:
            tprint_error(f"❌ Failed to load data for optimization")
            return None
    
    def validate_data_for_optimization(
        self,
        symbol: str,
        timeframe: str,
        pipeline_state: Dict[str, Any]
    ) -> bool:
        """
        Validate that data is available for optimization in the detected mode.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            pipeline_state: Pipeline state dictionary
            
        Returns:
            True if data is available, False otherwise
        """
        mode = self.detect_execution_mode(pipeline_state)
        
        tprint(f"🔍 Validating data availability for optimization")
        tprint_info(f"   → Symbol: {symbol}")
        tprint_info(f"   → Timeframe: {timeframe}")
        tprint_info(f"   → Mode: {mode.upper()}")
        
        return self.data_loader.validate_data_availability(symbol, timeframe, mode)
    
    def get_optimization_parameters(
        self,
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get optimization parameters based on detected execution mode.
        
        Args:
            pipeline_state: Pipeline state dictionary
            
        Returns:
            Dictionary with optimization parameters
        """
        mode = self.detect_execution_mode(pipeline_state)
        
        # Get mode configuration
        if mode == "light":
            config = get_light_mode_config()
        elif mode == "blank":
            config = get_blank_mode_config()
        else:  # full
            config = get_full_mode_config()
        
        # Get date range
        start_date, end_date = get_ares_mode_dates(mode)
        
        parameters = {
            'mode': mode,
            'lookback_days': config.lookback_days,
            'start_date': start_date,
            'end_date': end_date,
            'intensity_percentage': config.intensity_percentage,
            'computational_intensity': config.computational_intensity,
            'estimated_duration_minutes': config.estimated_duration_minutes,
            'max_trials': config.max_trials,
            'n_trials': config.n_trials,
            'enable_parallelization': config.enable_parallelization,
            'enable_caching': config.enable_caching
        }
        
        tprint_info(f"📊 Optimization parameters for {mode.upper()} mode:")
        tprint_info(f"   → Lookback days: {parameters['lookback_days']}")
        tprint_info(f"   → Date range: {start_date.date()} to {end_date.date()}")
        tprint_info(f"   → Intensity: {parameters['intensity_percentage']:.1%}")
        tprint_info(f"   → Max trials: {parameters['max_trials']}")
        
        return parameters
    
    def _get_mode_lookback_days(self, mode: str) -> int:
        """Get lookback days for a specific mode."""
        if mode == "light":
            return get_light_mode_config().lookback_days
        elif mode == "blank":
            return get_blank_mode_config().lookback_days
        else:  # full
            return get_full_mode_config().lookback_days
    
    def print_mode_summary(self, pipeline_state: Dict[str, Any]):
        """Print a summary of the detected mode and its parameters."""
        mode = self.detect_execution_mode(pipeline_state)
        parameters = self.get_optimization_parameters(pipeline_state)
        
        tprint(f"\n📊 ARES LAUNCHER MODE SUMMARY")
        tprint(f"=" * 50)
        tprint(f"Mode: {mode.upper()}")
        tprint(f"Description: {parameters.get('computational_intensity', 'Unknown')} intensity")
        tprint(f"Lookback Days: {parameters['lookback_days']}")
        tprint(f"Date Range: {parameters['start_date'].date()} to {parameters['end_date'].date()}")
        tprint(f"Intensity: {parameters['intensity_percentage']:.1%}")
        tprint(f"Max Trials: {parameters['max_trials']}")
        tprint(f"Parallelization: {parameters['enable_parallelization']}")
        tprint(f"Caching: {parameters['enable_caching']}")
        tprint(f"=" * 50)


# Convenience functions for easy integration
def load_data_for_feature_optimization(
    symbol: str,
    timeframe: str,
    pipeline_state: Dict[str, Any],
    data_dir: str = "historical_data",
    **kwargs
) -> Optional[Any]:
    """
    Convenience function to load data for feature optimization.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        pipeline_state: Pipeline state dictionary
        data_dir: Data directory
        **kwargs: Additional arguments
        
    Returns:
        Loaded DataFrame or None
    """
    optimizer = AresLauncherFeatureLookbackOptimizer(data_dir)
    return optimizer.load_data_for_optimization(symbol, timeframe, pipeline_state, **kwargs)


async def load_data_async_for_feature_optimization(
    symbol: str,
    timeframe: str,
    pipeline_state: Dict[str, Any],
    data_dir: str = "historical_data",
    **kwargs
) -> Optional[Any]:
    """
    Convenience function to asynchronously load data for feature optimization.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        pipeline_state: Pipeline state dictionary
        data_dir: Data directory
        **kwargs: Additional arguments
        
    Returns:
        Loaded DataFrame or None
    """
    optimizer = AresLauncherFeatureLookbackOptimizer(data_dir)
    return await optimizer.load_data_async_for_optimization(symbol, timeframe, pipeline_state, **kwargs)


def validate_data_for_feature_optimization(
    symbol: str,
    timeframe: str,
    pipeline_state: Dict[str, Any],
    data_dir: str = "historical_data"
) -> bool:
    """
    Convenience function to validate data for feature optimization.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        pipeline_state: Pipeline state dictionary
        data_dir: Data directory
        
    Returns:
        True if data is available, False otherwise
    """
    optimizer = AresLauncherFeatureLookbackOptimizer(data_dir)
    return optimizer.validate_data_for_optimization(symbol, timeframe, pipeline_state)


# Example usage
if __name__ == "__main__":
    async def main():
        # Example pipeline states for different modes
        pipeline_states = [
            {
                'execution_mode': 'light',
                'symbol': 'ETHUSDT',
                'timeframe': '15m'
            },
            {
                'execution_mode': 'blank',
                'symbol': 'ETHUSDT',
                'timeframe': '15m'
            },
            {
                'execution_mode': 'full',
                'symbol': 'ETHUSDT',
                'timeframe': '15m'
            }
        ]
        
        optimizer = AresLauncherFeatureLookbackOptimizer()
        
        for pipeline_state in pipeline_states:
            tprint(f"\n🧪 Testing {pipeline_state['execution_mode'].upper()} mode")
            
            # Print mode summary
            optimizer.print_mode_summary(pipeline_state)
            
            # Validate data availability
            is_available = optimizer.validate_data_for_optimization(
                pipeline_state['symbol'],
                pipeline_state['timeframe'],
                pipeline_state
            )
            print(f"Data available: {is_available}")
            
            # Load data
            data = optimizer.load_data_for_optimization(
                pipeline_state['symbol'],
                pipeline_state['timeframe'],
                pipeline_state
            )
            print(f"Data loaded: {data is not None}")
            if data is not None:
                print(f"Data shape: {data.shape}")
                print(f"Data mode: {data.attrs.get('ares_mode', 'Unknown')}")
                print(f"Lookback days: {data.attrs.get('lookback_days', 'Unknown')}")
    
    asyncio.run(main())