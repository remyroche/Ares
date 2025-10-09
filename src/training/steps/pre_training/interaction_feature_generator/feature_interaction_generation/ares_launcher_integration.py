"""
Ares Launcher Integration for Interactive Feature Generation

This module ensures that the interactive feature generation component
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


class AresLauncherInteractiveFeatureGenerator:
    """
    Interactive feature generator that respects ares_launcher configuration.
    
    This class ensures that the 20-day lookback period from "light" mode
    is properly applied when loading data for interactive feature generation.
    """
    
    def __init__(self, data_dir: str = "historical_data"):
        """Initialize the ares launcher interactive feature generator."""
        self.data_loader = AresLauncherDataLoader(data_dir)
        self.data_dir = data_dir
        
        tprint("🚀 Initializing Ares Launcher Interactive Feature Generator")
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
    
    def load_data_for_generation(
        self,
        symbol: str,
        timeframe: str,
        pipeline_state: Dict[str, Any],
        custom_start_date: Optional[datetime] = None,
        custom_end_date: Optional[datetime] = None
    ) -> Optional[Any]:
        """
        Load data for interactive feature generation respecting ares_launcher mode.
        
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
        
        tprint(f"📊 Loading data for interactive feature generation")
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
            tprint_success(f"✅ Data loaded successfully for feature generation")
            tprint_info(f"   → Records: {len(data)}")
            tprint_info(f"   → Date range: {data.index.min().date()} to {data.index.max().date()}")
            
            # Add mode information to data
            data.attrs['ares_mode'] = mode
            data.attrs['lookback_days'] = self._get_mode_lookback_days(mode)
            
            return data
        else:
            tprint_error(f"❌ Failed to load data for feature generation")
            return None
    
    async def load_data_async_for_generation(
        self,
        symbol: str,
        timeframe: str,
        pipeline_state: Dict[str, Any],
        custom_start_date: Optional[datetime] = None,
        custom_end_date: Optional[datetime] = None
    ) -> Optional[Any]:
        """
        Asynchronously load data for interactive feature generation.
        
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
        
        tprint(f"📊 Loading data asynchronously for interactive feature generation")
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
            tprint_success(f"✅ Data loaded successfully for feature generation")
            tprint_info(f"   → Records: {len(data)}")
            tprint_info(f"   → Date range: {data.index.min().date()} to {data.index.max().date()}")
            
            # Add mode information to data
            data.attrs['ares_mode'] = mode
            data.attrs['lookback_days'] = self._get_mode_lookback_days(mode)
            
            return data
        else:
            tprint_error(f"❌ Failed to load data for feature generation")
            return None
    
    def validate_data_for_generation(
        self,
        symbol: str,
        timeframe: str,
        pipeline_state: Dict[str, Any]
    ) -> bool:
        """
        Validate that data is available for generation in the detected mode.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            pipeline_state: Pipeline state dictionary
            
        Returns:
            True if data is available, False otherwise
        """
        mode = self.detect_execution_mode(pipeline_state)
        
        tprint(f"🔍 Validating data availability for feature generation")
        tprint_info(f"   → Symbol: {symbol}")
        tprint_info(f"   → Timeframe: {timeframe}")
        tprint_info(f"   → Mode: {mode.upper()}")
        
        return self.data_loader.validate_data_availability(symbol, timeframe, mode)
    
    def get_generation_parameters(
        self,
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get generation parameters based on detected execution mode.
        
        Args:
            pipeline_state: Pipeline state dictionary
            
        Returns:
            Dictionary with generation parameters
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
        
        # Adjust feature generation parameters based on mode
        if mode == "light":
            # Light mode: fewer features, faster generation
            feature_budget_pre = 60
            feature_budget_post = (15, 30)
            interactions_cap = 8
            transforms_per_parent = 1
            max_workers = 4
            batch_size = 1000
        elif mode == "blank":
            # Blank mode: moderate features
            feature_budget_pre = 100
            feature_budget_post = (25, 50)
            interactions_cap = 12
            transforms_per_parent = 1
            max_workers = 6
            batch_size = 1500
        else:  # full
            # Full mode: maximum features
            feature_budget_pre = 150
            feature_budget_post = (40, 80)
            interactions_cap = 20
            transforms_per_parent = 2
            max_workers = 8
            batch_size = 2000
        
        parameters = {
            'mode': mode,
            'lookback_days': config.lookback_days,
            'start_date': start_date,
            'end_date': end_date,
            'intensity_percentage': config.intensity_percentage,
            'computational_intensity': config.computational_intensity,
            'estimated_duration_minutes': config.estimated_duration_minutes,
            'feature_budget_pre': feature_budget_pre,
            'feature_budget_post': feature_budget_post,
            'interactions_cap': interactions_cap,
            'transforms_per_parent': transforms_per_parent,
            'max_workers': max_workers,
            'batch_size': batch_size,
            'enable_parallelization': config.enable_parallelization,
            'enable_caching': config.enable_caching
        }
        
        tprint_info(f"📊 Generation parameters for {mode.upper()} mode:")
        tprint_info(f"   → Lookback days: {parameters['lookback_days']}")
        tprint_info(f"   → Date range: {start_date.date()} to {end_date.date()}")
        tprint_info(f"   → Intensity: {parameters['intensity_percentage']:.1%}")
        tprint_info(f"   → Feature budget (pre): {parameters['feature_budget_pre']}")
        tprint_info(f"   → Feature budget (post): {parameters['feature_budget_post']}")
        tprint_info(f"   → Interactions cap: {parameters['interactions_cap']}")
        tprint_info(f"   → Max workers: {parameters['max_workers']}")
        
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
        parameters = self.get_generation_parameters(pipeline_state)
        
        tprint(f"\n📊 ARES LAUNCHER MODE SUMMARY")
        tprint(f"=" * 50)
        tprint(f"Mode: {mode.upper()}")
        tprint(f"Description: {parameters.get('computational_intensity', 'Unknown')} intensity")
        tprint(f"Lookback Days: {parameters['lookback_days']}")
        tprint(f"Date Range: {parameters['start_date'].date()} to {parameters['end_date'].date()}")
        tprint(f"Intensity: {parameters['intensity_percentage']:.1%}")
        tprint(f"Feature Budget (pre): {parameters['feature_budget_pre']}")
        tprint(f"Feature Budget (post): {parameters['feature_budget_post']}")
        tprint(f"Interactions Cap: {parameters['interactions_cap']}")
        tprint(f"Max Workers: {parameters['max_workers']}")
        tprint(f"Batch Size: {parameters['batch_size']}")
        tprint(f"Parallelization: {parameters['enable_parallelization']}")
        tprint(f"Caching: {parameters['enable_caching']}")
        tprint(f"=" * 50)


# Convenience functions for easy integration
def load_data_for_interactive_generation(
    symbol: str,
    timeframe: str,
    pipeline_state: Dict[str, Any],
    data_dir: str = "historical_data",
    **kwargs
) -> Optional[Any]:
    """
    Convenience function to load data for interactive feature generation.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        pipeline_state: Pipeline state dictionary
        data_dir: Data directory
        **kwargs: Additional arguments
        
    Returns:
        Loaded DataFrame or None
    """
    generator = AresLauncherInteractiveFeatureGenerator(data_dir)
    return generator.load_data_for_generation(symbol, timeframe, pipeline_state, **kwargs)


async def load_data_async_for_interactive_generation(
    symbol: str,
    timeframe: str,
    pipeline_state: Dict[str, Any],
    data_dir: str = "historical_data",
    **kwargs
) -> Optional[Any]:
    """
    Convenience function to asynchronously load data for interactive feature generation.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        pipeline_state: Pipeline state dictionary
        data_dir: Data directory
        **kwargs: Additional arguments
        
    Returns:
        Loaded DataFrame or None
    """
    generator = AresLauncherInteractiveFeatureGenerator(data_dir)
    return await generator.load_data_async_for_generation(symbol, timeframe, pipeline_state, **kwargs)


def validate_data_for_interactive_generation(
    symbol: str,
    timeframe: str,
    pipeline_state: Dict[str, Any],
    data_dir: str = "historical_data"
) -> bool:
    """
    Convenience function to validate data for interactive feature generation.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        pipeline_state: Pipeline state dictionary
        data_dir: Data directory
        
    Returns:
        True if data is available, False otherwise
    """
    generator = AresLauncherInteractiveFeatureGenerator(data_dir)
    return generator.validate_data_for_generation(symbol, timeframe, pipeline_state)


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
        
        generator = AresLauncherInteractiveFeatureGenerator()
        
        for pipeline_state in pipeline_states:
            tprint(f"\n🧪 Testing {pipeline_state['execution_mode'].upper()} mode")
            
            # Print mode summary
            generator.print_mode_summary(pipeline_state)
            
            # Validate data availability
            is_available = generator.validate_data_for_generation(
                pipeline_state['symbol'],
                pipeline_state['timeframe'],
                pipeline_state
            )
            print(f"Data available: {is_available}")
            
            # Load data
            data = generator.load_data_for_generation(
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