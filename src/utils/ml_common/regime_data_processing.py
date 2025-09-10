"""
Regime Data Processing Utilities

This module provides comprehensive regime data processing capabilities.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Callable
import logging
import time
from dataclasses import dataclass
from enum import Enum

from ..math_validation import safe_divide
from ..common_operations import create_fallback_logger
from ..m1_memory_optimizer import get_m1_memory_optimizer
from ..parquet_utils import ParquetUtils

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"

@dataclass
class ProcessingStats:
    files_processed: int = 0
    total_rows_processed: int = 0
    processing_time: float = 0.0
    error_count: int = 0

class EnhancedRegimeDataProcessor:
    def __init__(self, processing_mode: ProcessingMode = ProcessingMode.SYNC):
        self.processing_mode = processing_mode
        self.logger = create_fallback_logger("EnhancedRegimeDataProcessor")
        self.memory_optimizer = get_m1_memory_optimizer()
        self.parquet_utils = ParquetUtils()
        self.stats = ProcessingStats()

    def process_regime_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process regime data with optimization."""
        start_time = time.time()
        
        try:
            # Optimize data types
            optimized_data = self._optimize_data_types(data)
            
            # Update statistics
            self.stats.total_rows_processed += len(optimized_data)
            self.stats.processing_time += time.time() - start_time
            
            return optimized_data
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            self.stats.error_count += 1
            raise

    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        optimized_data = data.copy()
        
        # Optimize numeric columns
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            col_data = optimized_data[col]
            
            if col_data.dtype == 'float64':
                if col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                    optimized_data[col] = col_data.astype(np.float32)
            elif col_data.dtype == 'int64':
                if col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                    optimized_data[col] = col_data.astype(np.int32)
        
        return optimized_data

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'total_rows_processed': self.stats.total_rows_processed,
            'processing_time': self.stats.processing_time,
            'error_count': self.stats.error_count,
            'rows_per_second': safe_divide(
                self.stats.total_rows_processed,
                self.stats.processing_time
            )
        }

# Global instance
enhanced_regime_data_processor = EnhancedRegimeDataProcessor()
RegimeDataProcessor = EnhancedRegimeDataProcessor