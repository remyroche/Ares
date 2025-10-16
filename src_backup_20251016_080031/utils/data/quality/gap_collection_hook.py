"""
Gap Collection Hook

This module provides functionality to hook with data collection pipelines
when large gaps are detected, triggering re-downloading of missing data.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from src.utils.logger import system_logger

logger = system_logger.getChild('GapCollectionHook')

class GapCollectionStrategy(Enum):
    """Strategies for handling large gaps."""
    IMMEDIATE_DOWNLOAD = "immediate_download"
    SCHEDULED_DOWNLOAD = "scheduled_download"
    MANUAL_REVIEW = "manual_review"
    IGNORE = "ignore"

class GapCollectionHook:
    """Hook for triggering data collection when large gaps are detected."""
    
    def __init__(self, strategy: GapCollectionStrategy = GapCollectionStrategy.IMMEDIATE_DOWNLOAD):
        """Initialize gap collection hook.
        
        Args:
            strategy: Strategy for handling large gaps
        """
        self.logger = logger.getChild('GapCollectionHook')
        self.strategy = strategy
        self.collection_history = []
        
    def should_trigger_collection(self, gap_info: Dict[str, Any], data_type: str) -> bool:
        """Determine if data collection should be triggered for a gap.
        
        Args:
            gap_info: Information about the gap
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            
        Returns:
            True if collection should be triggered
        """
        gap_size = gap_info.get('gap_size', 0)
        gap_type = gap_info.get('gap_type', 'unknown')
        
        # Data-type specific thresholds for triggering collection
        collection_thresholds = {
            'aggtrades': 0.5,  # 0.5 seconds
            'klines': 66,      # 1.1 minutes for 1m klines (avoid unnecessary downloads)
            'futures': 32400   # 9 hours
        }
        
        threshold = collection_thresholds.get(data_type, 300)  # Default 5 minutes
        
        if gap_size >= threshold:
            self.logger.warning(f"🔄 LARGE GAP DETECTED: {gap_size}s >= {threshold}s threshold for {data_type}")
            self.logger.warning(f"   Gap type: {gap_type}")
            return True
            
        return False
    
    def trigger_data_collection(self, gap_info: Dict[str, Any], data_type: str, 
                              symbol: str, exchange: str, timeframe: str = None) -> Dict[str, Any]:
        """Trigger data collection for a specific gap.
        
        Args:
            gap_info: Information about the gap
            data_type: Type of data to collect
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe (for klines data)
            
        Returns:
            Collection result information
        """
        if not self.should_trigger_collection(gap_info, data_type):
            return {'triggered': False, 'reason': 'Gap below threshold'}
        
        self.logger.info(f"🔄 Triggering data collection for {data_type} gap")
        self.logger.info(f"   Symbol: {symbol}, Exchange: {exchange}")
        self.logger.info(f"   Gap: {gap_info}")
        
        try:
            # Import data collection components
            from src.training.steps.data_collection.sub_pipeline import DataCollectionSubPipeline, SubPipelineConfig, ExecutionMode
            
            # Create collection configuration
            config = SubPipelineConfig(
                mode=ExecutionMode.FULL,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe or '1m',
                data_dir='data/training'
            )
            
            # Create and execute data collection pipeline
            collection_pipeline = DataCollectionSubPipeline(config)
            
            # For now, just log the collection attempt
            # In a full implementation, this would actually trigger the collection
            self.logger.info("📡 Data collection pipeline would be triggered here")
            self.logger.info("   This would download missing data for the gap period")
            
            # Record collection attempt
            collection_record = {
                'timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'gap_info': gap_info,
                'strategy': self.strategy.value,
                'status': 'attempted'
            }
            self.collection_history.append(collection_record)
            
            return {
                'triggered': True,
                'status': 'attempted',
                'collection_record': collection_record
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to trigger data collection: {e}")
            return {
                'triggered': False,
                'error': str(e)
            }
    
    def get_collection_report(self) -> Dict[str, Any]:
        """Get report of collection attempts."""
        if not self.collection_history:
            return {
                'timestamp': datetime.now().isoformat(),
                'total_attempts': 0,
                'attempts': []
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_attempts': len(self.collection_history),
            'attempts_by_data_type': self._count_by_field('data_type'),
            'attempts_by_exchange': self._count_by_field('exchange'),
            'attempts_by_strategy': self._count_by_field('strategy'),
            'recent_attempts': self.collection_history[-10:],
            'attempts': self.collection_history
        }
    
    def _count_by_field(self, field: str) -> Dict[str, int]:
        """Count attempts by a specific field."""
        counts = {}
        for record in self.collection_history:
            value = record.get(field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts

# Global instance for easy access
_gap_collection_hook: Optional[GapCollectionHook] = None

def get_gap_collection_hook(strategy: GapCollectionStrategy = GapCollectionStrategy.IMMEDIATE_DOWNLOAD) -> GapCollectionHook:
    """Get the global gap collection hook instance.
    
    Args:
        strategy: Strategy for handling large gaps
        
    Returns:
        GapCollectionHook instance
    """
    global _gap_collection_hook
    if _gap_collection_hook is None:
        _gap_collection_hook = GapCollectionHook(strategy)
    return _gap_collection_hook

def trigger_gap_collection(gap_info: Dict[str, Any], data_type: str, 
                          symbol: str, exchange: str, timeframe: str = None) -> Dict[str, Any]:
    """Convenience function to trigger gap collection.
    
    Args:
        gap_info: Information about the gap
        data_type: Type of data to collect
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe (for klines data)
        
    Returns:
        Collection result information
    """
    hook = get_gap_collection_hook()
    return hook.trigger_data_collection(gap_info, data_type, symbol, exchange, timeframe)