from typing import Callable
from typing import Any
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

'Decorator for automatic per-regime processing in training steps.\n\nThis module provides a decorator that automatically handles per-regime processing\nfor training steps, ensuring consistent regime-based execution across steps 4-21.\n'
import functools
from src.training.steps.market_analysis.regime_handler import regime_handler
import pandas as pd

from src.utils.common_operations import get_logger
import logging
import numpy as np

logger = get_logger('RegimeProcessingDecorator')

def per_regime_processing(result_type: str='generic', parallel: bool = True, preserve_context: bool = True, context_window: int = 100) -> None:
    """Decorator to automatically process functions on a per-regime basis.
    
    This decorator wraps a processing function to automatically:
    1. Load unified regime data
    2. Process each regime separately
    3. Save per-regime results
    4. Return aggregated results
    
    Args:
        result_type: Type of results for file naming
        parallel: Whether to process regimes in parallel
        preserve_context: Whether to preserve temporal context
        context_window: Number of rows for context preservation
        
    The decorated function should have the signature:
        async def process_func(data: pd.DataFrame, regime_id: int, **kwargs) -> Any
    """

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        async def wrapper(symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> Dict[int, Any]:
            """Wrapper that handles per-regime processing."""
            logger.info(f'🔄 Starting per-regime processing for {func.__name__}')
            data = await regime_handler.load_unified_regime_data(symbol, exchange, timeframe, data_dir)
            if data is None or data.empty:
                logger.error('❌ No regime data found')
                return {}

            async def regime_processor(regime_data: pd.DataFrame, **proc_kwargs) -> Any:
                if 'is_regime_context' in regime_data.columns:
                    context_mask = regime_data['is_regime_context']
                    regime_data = regime_data.drop(columns=['is_regime_context'])
                    proc_kwargs['context_mask'] = context_mask
                return await func(regime_data, **proc_kwargs)
            results = await regime_handler.process_per_regime(data = data, processing_func = regime_processor, symbol = symbol, exchange = exchange, timeframe = timeframe, parallel = parallel, **kwargs)
            step_name = func.__name__.replace('process_', '').replace('_regime', '')
            await regime_handler.save_regime_results(results = results, step_name = step_name, symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, result_type = result_type)
            logger.info(f'✅ Completed per-regime processing for {func.__name__}')
            return results
        wrapper.__name__ = f'per_regime_{func.__name__}'
        wrapper.__doc__ = f'Per-regime version of {func.__name__}\n\n{func.__doc__}'
        return wrapper
    return decorator

def aggregate_regime_results(results: Dict[int, pd.DataFrame], aggregation_method: str='concat') -> pd.DataFrame:
    """Aggregate per-regime results into a single DataFrame.
    
    Args:
        results: Dictionary mapping regime IDs to DataFrames
        aggregation_method: Method to use ('concat', 'merge', 'average')
        
    Returns:
        Aggregated DataFrame
    """
    if not results:
        return pd.DataFrame()
    valid_results = {k: v for k, v in results.items() if v is not None and (not v.empty)}
    if not valid_results:
        return pd.DataFrame()
    if aggregation_method == 'concat':
        dfs = []
        for regime_id, df in valid_results.items():
            df_copy = df.copy()
            df_copy['regime_id'] = regime_id
            dfs.append(df_copy)
        return pd.concat(dfs, ignore_index = True)
    elif aggregation_method == 'merge':
        base_df = None
        for regime_id, df in valid_results.items():
            df_copy = df.copy()
            df_copy.columns = [f'{col}_regime_{regime_id}' if col != 'timestamp' else col for col in df_copy.columns]
            if base_df is None:
                base_df = df_copy
            else:
                base_df = pd.merge(base_df, df_copy, on='timestamp', how='outer')
        return base_df
    elif aggregation_method == 'average':
        numeric_cols = valid_results[list(valid_results.keys())[0]].select_dtypes(include=['number']).columns
        avg_df = pd.DataFrame()
        for col in numeric_cols:
            values = []
            for df in valid_results.values():
                if col in df.columns:
                    values.append(df[col])
            if values:
                avg_df[col] = pd.concat(values).groupby(level = 0).mean()
        return avg_df
    else:
        raise ValueError(f'Unknown aggregation method: {aggregation_method}')

class RegimeProcessingContext:
    """Context manager for regime-specific processing.
    
    This provides a clean way to handle regime processing with proper
    setup and teardown, useful for steps that need custom regime handling.
    """
    @log_important_calls

    def __init__(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.data_dir = data_dir
        self.regime_data = None
        self.regime_ids = None

    async def __aenter__(self) -> None:
        """Load regime data on entry."""
        self.regime_data = await regime_handler.load_unified_regime_data(self.symbol, self.exchange, self.timeframe, self.data_dir)
        if self.regime_data is not None:
            self.regime_ids = regime_handler.get_regime_ids(self.regime_data)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Cleanup on exit."""
        pass

    def get_regime_data(self, regime_id: int, preserve_context: bool = True) -> pd.DataFrame:
        """Get data for a specific regime."""
        if self.regime_data is None:
            return pd.DataFrame()
        return regime_handler.filter_data_by_regime(self.regime_data, regime_id, preserve_context)

    async def process_regime(self, regime_id: int, processing_func: Callable, **kwargs) -> Any:
        """Process a specific regime."""
        regime_data = self.get_regime_data(regime_id)
        if regime_data.empty:
            logger.warning(f'⚠️ No data for regime {regime_id}')
            return None
        return await processing_func(regime_data, regime_id = regime_id, symbol = self.symbol, exchange = self.exchange, timeframe = self.timeframe, **kwargs)