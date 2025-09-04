"""Unified Regime Handler for Consistent Per-HMM Regime Data Processing.

This module provides a centralized way to handle regime data across all training steps,
ensuring that steps 4-21 perform tasks on a per-HMM regime basis with consistent methods.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from functools import lru_cache
import logging

from src.utils.logger import getChild as get_logger
from src.utils.common_operations import ensure_directory, safe_json_dump, safe_json_load
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, cached, validates, handles_errors, log_execution_time
from src.core.decorators.errors import handles_errors


logger = get_logger('RegimeHandler')


class RegimeHandler:
    """Unified handler for regime-specific data operations across all training steps."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the regime handler.
        
        Args:
            config: Configuration dictionary containing regime processing parameters
        """
        self.config = config or {}
        self.logger = get_logger('RegimeHandler')
        self.standards = pipeline_standards
        self._cached_regime_data = {}
        self._regime_metadata = {}
        
    @traced(span_name='load_unified_regime_data')
    @cached
    async def load_unified_regime_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load the unified regime dataset created by Step 4.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            
        Returns:
            DataFrame with unified regime data or None if not found
        """
        try:
            training_dir = Path(data_dir) / 'training'
            unified_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            
            if not unified_file.exists():
                self.logger.error(f"❌ Unified regime data not found: {unified_file}")
                return None
                
            data = pd.read_parquet(unified_file)
            self.logger.info(f"✅ Loaded unified regime data: {len(data)} rows from {unified_file}")
            
            # Cache metadata for faster access
            cache_key = f"{exchange}_{symbol}_{timeframe}"
            self._cached_regime_data[cache_key] = data
            
            # Load regime metadata
            metadata_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_metadata.json'
            if metadata_file.exists():
                self._regime_metadata[cache_key] = safe_json_load(metadata_file)
                
            return data
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading unified regime data: {e}")
            return None
    
    @traced(span_name='get_regime_ids')
    def get_regime_ids(self, data: pd.DataFrame) -> List[int]:
        """Get unique regime IDs from the data.
        
        Args:
            data: DataFrame with composite_cluster_id column
            
        Returns:
            List of unique regime IDs
        """
        if 'composite_cluster_id' not in data.columns:
            self.logger.error("❌ No composite_cluster_id column found in data")
            return []
            
        regime_ids = sorted(data['composite_cluster_id'].unique())
        self.logger.info(f"📊 Found {len(regime_ids)} unique regimes: {regime_ids}")
        return regime_ids
    
    @traced(span_name='filter_data_by_regime')
    def filter_data_by_regime(
        self, 
        data: pd.DataFrame, 
        regime_id: int,
        preserve_context: bool = True,
        context_window: int = 100
    ) -> pd.DataFrame:
        """Filter data for a specific regime.
        
        Args:
            data: DataFrame with regime data
            regime_id: Regime ID to filter for
            preserve_context: Whether to preserve temporal context around regime periods
            context_window: Number of rows before/after regime transitions to include
            
        Returns:
            Filtered DataFrame for the specified regime
        """
        if 'composite_cluster_id' not in data.columns:
            self.logger.error("❌ No composite_cluster_id column found in data")
            return pd.DataFrame()
            
        if preserve_context:
            # Find regime boundaries
            regime_mask = data['composite_cluster_id'] == regime_id
            regime_changes = regime_mask.ne(regime_mask.shift())
            regime_starts = data.index[regime_changes & regime_mask].tolist()
            regime_ends = data.index[regime_changes & ~regime_mask].tolist()
            
            # Create extended mask with context
            extended_mask = pd.Series(False, index=data.index)
            
            for start_idx in regime_starts:
                # Include context before regime start
                context_start = max(0, start_idx - context_window)
                
                # Find corresponding end
                end_idx = None
                for end in regime_ends:
                    if end > start_idx:
                        end_idx = end
                        break
                        
                if end_idx is None:
                    end_idx = len(data)
                    
                # Include context after regime end
                context_end = min(len(data), end_idx + context_window)
                
                extended_mask.iloc[context_start:context_end] = True
            
            regime_data = data[extended_mask].copy()
            
            # Mark context rows
            regime_data['is_regime_context'] = ~(regime_data['composite_cluster_id'] == regime_id)
            
            self.logger.info(
                f"📊 Filtered regime {regime_id} data with context: "
                f"{len(regime_data)} rows ({(~regime_data['is_regime_context']).sum()} regime, "
                f"{regime_data['is_regime_context'].sum()} context)"
            )
        else:
            # Simple filtering without context
            regime_data = data[data['composite_cluster_id'] == regime_id].copy()
            regime_data['is_regime_context'] = False
            
            self.logger.info(f"📊 Filtered regime {regime_id} data: {len(regime_data)} rows")
            
        return regime_data
    
    @traced(span_name='process_per_regime')
    @handles_errors
    async def process_per_regime(
        self,
        data: pd.DataFrame,
        processing_func: Callable,
        symbol: str,
        exchange: str,
        timeframe: str,
        parallel: bool = True,
        **kwargs
    ) -> Dict[int, Any]:
        """Process data for each regime using the provided function.
        
        Args:
            data: DataFrame with regime data
            processing_func: Async function to process regime data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            parallel: Whether to process regimes in parallel
            **kwargs: Additional arguments to pass to processing_func
            
        Returns:
            Dictionary mapping regime IDs to processing results
        """
        regime_ids = self.get_regime_ids(data)
        if not regime_ids:
            self.logger.error("❌ No regimes found in data")
            return {}
            
        results = {}
        
        if parallel:
            # Process regimes in parallel
            tasks = []
            for regime_id in regime_ids:
                regime_data = self.filter_data_by_regime(data, regime_id)
                task = asyncio.create_task(
                    self._process_single_regime(
                        regime_id, regime_data, processing_func, 
                        symbol, exchange, timeframe, **kwargs
                    )
                )
                tasks.append((regime_id, task))
                
            # Gather results
            for regime_id, task in tasks:
                try:
                    result = await task
                    results[regime_id] = result
                except Exception as e:
                    self.logger.error(f"❌ Error processing regime {regime_id}: {e}")
                    results[regime_id] = None
        else:
            # Process regimes sequentially
            for regime_id in regime_ids:
                try:
                    regime_data = self.filter_data_by_regime(data, regime_id)
                    result = await self._process_single_regime(
                        regime_id, regime_data, processing_func,
                        symbol, exchange, timeframe, **kwargs
                    )
                    results[regime_id] = result
                except Exception as e:
                    self.logger.error(f"❌ Error processing regime {regime_id}: {e}")
                    results[regime_id] = None
                    
        successful_regimes = sum(1 for r in results.values() if r is not None)
        self.logger.info(
            f"✅ Processed {successful_regimes}/{len(regime_ids)} regimes successfully"
        )
        
        return results
    
    async def _process_single_regime(
        self,
        regime_id: int,
        regime_data: pd.DataFrame,
        processing_func: Callable,
        symbol: str,
        exchange: str,
        timeframe: str,
        **kwargs
    ) -> Any:
        """Process data for a single regime.
        
        Args:
            regime_id: Regime ID
            regime_data: Filtered data for the regime
            processing_func: Async function to process the data
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            **kwargs: Additional arguments
            
        Returns:
            Processing result
        """
        self.logger.info(f"🔄 Processing regime {regime_id} with {len(regime_data)} rows")
        
        # Add regime metadata to kwargs
        kwargs['regime_id'] = regime_id
        kwargs['symbol'] = symbol
        kwargs['exchange'] = exchange
        kwargs['timeframe'] = timeframe
        
        # Call the processing function
        result = await processing_func(regime_data, **kwargs)
        
        self.logger.info(f"✅ Completed processing for regime {regime_id}")
        return result
    
    @traced(span_name='save_regime_results')
    async def save_regime_results(
        self,
        results: Dict[int, Any],
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        result_type: str = 'generic'
    ) -> bool:
        """Save per-regime processing results.
        
        Args:
            results: Dictionary mapping regime IDs to results
            step_name: Name of the processing step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            result_type: Type of results (for file naming)
            
        Returns:
            Success status
        """
        try:
            output_dir = ensure_directory(Path(data_dir) / 'regime_results' / step_name)
            
            # Save individual regime results
            for regime_id, result in results.items():
                if result is None:
                    continue
                    
                # Determine file format based on result type
                if isinstance(result, pd.DataFrame):
                    filename = f"{exchange}_{symbol}_{timeframe}_regime_{regime_id}_{result_type}.parquet"
                    filepath = output_dir / filename
                    result.to_parquet(filepath, index=False)
                    self.logger.info(f"✅ Saved regime {regime_id} DataFrame: {filepath}")
                elif isinstance(result, dict):
                    filename = f"{exchange}_{symbol}_{timeframe}_regime_{regime_id}_{result_type}.json"
                    filepath = output_dir / filename
                    safe_json_dump(result, filepath, indent=2)
                    self.logger.info(f"✅ Saved regime {regime_id} JSON: {filepath}")
                else:
                    self.logger.warning(f"⚠️ Unsupported result type for regime {regime_id}: {type(result)}")
                    
            # Save summary metadata
            summary = {
                'step_name': step_name,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'total_regimes': len(results),
                'successful_regimes': sum(1 for r in results.values() if r is not None),
                'regime_ids': list(results.keys()),
                'result_type': result_type,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            summary_file = output_dir / f"{exchange}_{symbol}_{timeframe}_regime_processing_summary.json"
            safe_json_dump(summary, summary_file, indent=2)
            self.logger.info(f"✅ Saved regime processing summary: {summary_file}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving regime results: {e}")
            return False
    
    @traced(span_name='load_regime_results')
    @cached
    async def load_regime_results(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        result_type: str = 'generic'
    ) -> Dict[int, Any]:
        """Load previously saved per-regime results.
        
        Args:
            step_name: Name of the processing step
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            result_type: Type of results to load
            
        Returns:
            Dictionary mapping regime IDs to results
        """
        try:
            results_dir = Path(data_dir) / 'regime_results' / step_name
            if not results_dir.exists():
                self.logger.warning(f"⚠️ No results directory found: {results_dir}")
                return {}
                
            # Load summary to get regime IDs
            summary_file = results_dir / f"{exchange}_{symbol}_{timeframe}_regime_processing_summary.json"
            if not summary_file.exists():
                self.logger.warning(f"⚠️ No summary file found: {summary_file}")
                return {}
                
            summary = safe_json_load(summary_file)
            regime_ids = summary.get('regime_ids', [])
            
            results = {}
            for regime_id in regime_ids:
                # Try loading parquet first
                parquet_file = results_dir / f"{exchange}_{symbol}_{timeframe}_regime_{regime_id}_{result_type}.parquet"
                json_file = results_dir / f"{exchange}_{symbol}_{timeframe}_regime_{regime_id}_{result_type}.json"
                
                if parquet_file.exists():
                    results[regime_id] = pd.read_parquet(parquet_file)
                elif json_file.exists():
                    results[regime_id] = safe_json_load(json_file)
                else:
                    self.logger.warning(f"⚠️ No result file found for regime {regime_id}")
                    
            self.logger.info(f"✅ Loaded results for {len(results)}/{len(regime_ids)} regimes")
            return results
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading regime results: {e}")
            return {}
    
    def get_regime_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for each regime in the data.
        
        Args:
            data: DataFrame with regime data
            
        Returns:
            Dictionary with regime statistics
        """
        regime_ids = self.get_regime_ids(data)
        stats = {
            'total_regimes': len(regime_ids),
            'total_data_points': len(data),
            'regime_details': {}
        }
        
        for regime_id in regime_ids:
            regime_data = data[data['composite_cluster_id'] == regime_id]
            regime_stats = {
                'count': len(regime_data),
                'percentage': len(regime_data) / len(data) * 100,
                'date_range': {
                    'start': regime_data['timestamp'].min().isoformat(),
                    'end': regime_data['timestamp'].max().isoformat()
                }
            }
            
            # Add price statistics if available
            if 'close' in regime_data.columns:
                regime_stats['price_stats'] = {
                    'mean': float(regime_data['close'].mean()),
                    'std': float(regime_data['close'].std()),
                    'min': float(regime_data['close'].min()),
                    'max': float(regime_data['close'].max())
                }
                
            stats['regime_details'][f'regime_{regime_id}'] = regime_stats
            
        return stats


# Global instance for easy access
regime_handler = RegimeHandler()