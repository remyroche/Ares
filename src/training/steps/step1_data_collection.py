from ..standardized_parquet_handler import standardized_parquet_handler
"""
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

Step 1: Data Collection

This module handles the data collection step of the training pipeline.
It downloads and consolidates all required data for training.
"""
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Core imports
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)

# Import logging decorators
try:
    from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls
    LOGGING_DECORATORS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Logging decorators not available: {e}")
    LOGGING_DECORATORS_AVAILABLE = False
    
    def log_step_functions(func):
        """Fallback decorator when logging decorators are not available."""
        return func
    
    def log_important_calls(func):
        """Fallback decorator when logging decorators are not available."""
        return func

# Data collection imports with proper relative paths
try:
    from .data_collection.data_downloader import download_all_data_with_consolidation
    DATA_DOWNLOADER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data downloader not available: {e}")
    DATA_DOWNLOADER_AVAILABLE = False
    
    def download_all_data_with_consolidation(*args, **kwargs):
        """Fallback function when data downloader is not available."""
        logging.error("Data downloader not available - cannot download data")
        return False

# Data quality decorators import with proper relative path
try:
    from .data_collection.data_quality_components.validation_decorators import validate_data
    DATA_QUALITY_DECORATORS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Data quality decorators not available: {e}")
    DATA_QUALITY_DECORATORS_AVAILABLE = False
@log_step_functions 
def validate_data(func):
    """Fallback decorator when validation decorators are not available."""
    return func

class Step1DataCollection:
    """
    Step 1: Data Collection
    
    This class handles the data collection step of the training pipeline.
    It downloads and consolidates all required data for training.
    """
    @log_important_calls
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Step1 data collection.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('Step1DataCollection')
        self.data_dir = config.get('data_dir', 'data')
        self.symbols = config.get('symbols', [])
        self.exchanges = config.get('exchanges', [])
        self.intervals = config.get('intervals', ['1m'])
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info("Step1 Data Collection initialized")
    
    @handles_errors(fallback=False)
    async def collect_data(self) -> bool:
        """Collect data for all configured symbols and exchanges.
        
        Returns:
            bool: True if data collection successful, False otherwise
        """
        try:
            self.logger.info("Starting data collection...")
            
            # Initialize lookahead bias detector
            current_time = datetime.now()
            bias_detector = get_global_detector()
            bias_detector.set_current_timestamp(current_time)
            
            if not DATA_DOWNLOADER_AVAILABLE:
                self.logger.error("Data downloader not available - cannot collect data")
                return False
            
            success_count = 0
            total_tasks = len(self.symbols) * len(self.exchanges) * len(self.intervals)
            
            for symbol in self.symbols:
                for exchange in self.exchanges:
                    for interval in self.intervals:
                        try:
                            self.logger.info(f"Collecting data for {symbol} on {exchange} ({interval})")
                            
                            success = await download_all_data_with_consolidation(
                                symbol=symbol,
                                exchange_name=exchange,
                                interval=interval,
                                data_dir=self.data_dir
                            )
                            
                            if success:
                                success_count += 1
                                self.logger.info(f"✅ Successfully collected data for {symbol} on {exchange} ({interval})")
                            else:
                                self.logger.error(f"❌ Failed to collect data for {symbol} on {exchange} ({interval})")
                                
                        except Exception as e:
                            self.logger.exception(f"Error collecting data for {symbol} on {exchange} ({interval}): {e}")
            
            success_rate = success_count / total_tasks if total_tasks > 0 else 0
            self.logger.info(f"Data collection completed. Success rate: {success_rate:.2%} ({success_count}/{total_tasks})")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.exception(f"Error in data collection: {e}")
            return False
    
    @validate_data
    def validate_collected_data(self, data: pd.DataFrame) -> bool:
        """Validate collected data quality.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            if data.empty:
                self.logger.warning("Data is empty")
                return False
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for NaN values with safe division
            try:
                total_cells = len(data) * len(data.columns)
                nan_count = data.isnull().sum().sum()
                nan_ratio = safe_divide(nan_count, total_cells, 0.0)
                if nan_ratio > 0.1:  # More than 10% NaN values
                    self.logger.warning(f"High NaN ratio: {nan_ratio:.2%}")
                    return False
            except MathValidationError as e:
                self.logger.warning(f"Mathematical validation error in NaN ratio calculation: {e}")
                return False
            
            # Check for infinite values
            inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                self.logger.warning(f"Found {inf_count} infinite values")
                return False
            
            # Validate no lookahead bias in data
            try:
                if hasattr(data, 'index') and len(data) > 0:
                    current_time = data.index[-1] if hasattr(data.index, '__getitem__') else None
                    if current_time:
                        bias_detector = get_global_detector()
                        bias_detector.set_current_timestamp(current_time)
                        data = validate_no_future_data(data, 'timestamp', current_time)
                        self.logger.info("✅ Lookahead bias validation passed")
            except LookaheadBiasError as e:
                self.logger.error(f"Lookahead bias detected: {e}")
                return False
            except Exception as e:
                self.logger.warning(f"Lookahead bias validation failed: {e}")
            
            self.logger.info("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error validating data: {e}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data.
        
        Returns:
            Dict containing data summary information
        """
        try:
            summary = {
                'data_dir': self.data_dir,
                'symbols': self.symbols,
                'exchanges': self.exchanges,
                'intervals': self.intervals,
                'timestamp': datetime.now().isoformat()
            }
            
            # Count files in data directory
            data_path = Path(self.data_dir)
            if data_path.exists():
                data_files = list(data_path.glob('**/*.parquet')) + list(data_path.glob('**/*.csv'))
                summary['data_files_count'] = len(data_files)
                summary['data_files'] = [str(f) for f in data_files]
            else:
                summary['data_files_count'] = 0
                summary['data_files'] = []
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"Error getting data summary: {e}")
            return {'error': str(e)}

# Main execution function
async def main():
    """Main execution function for Step1 data collection."""
    # Example configuration
    config = {
        'data_dir': 'data',
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'exchanges': ['binance'],
        'intervals': ['1m', '5m']
    }
    
    # Create and run data collection
    collector = Step1DataCollection(config)
    success = await collector.collect_data()
    
    if success:
        print("✅ Data collection completed successfully")
        summary = collector.get_data_summary()
        print(f"Data summary: {summary}")
    else:
        print("❌ Data collection failed")

if __name__ == "__main__":
    asyncio.run(main())