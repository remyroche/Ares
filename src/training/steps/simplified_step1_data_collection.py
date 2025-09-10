"""
Simplified Step 1: Data Collection

This module provides a simplified version of step1_data_collection using the new
infrastructure with MLPipelineOrchestrator and utility-based approaches.

Key Features:
- Uses SimplifiedPipelineManager for execution and monitoring
- Uses ConfigurationValidator for standardized config validation
- Uses DataQualityUtilities for unified data validation
- Simple function-based approach instead of complex class
- Automatic error handling and recovery
- Comprehensive logging and monitoring
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# Import new simplified infrastructure
from .simplified_pipeline_infrastructure import (
    SimplifiedPipelineManager,
    create_simple_step_function,
    create_data_processing_step_function
)

# Import standardized validation
from .standardized_config_validation import (
    validate_config,
    validate_and_fix_config
)

# Import unified data quality
from .unified_data_quality import (
    validate_data_quality,
    clean_data,
    generate_quality_report
)

# Import step06 utilities for dependency injection
from src.utils.step06_utilities import (
    Step06UtilityContainer,
    get_utility_container
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


# Simplified data collection step function
async def step1_data_collection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified data collection logic using utilities.
    
    Args:
        config: Configuration dictionary
        pipeline_state: Current pipeline state
        
    Returns:
        Data collection result
    """
    logger.info("📊 Starting simplified data collection...")
    
    try:
        # Get utility container for dependency injection
        utility_container = get_utility_container(config)
        
        # Extract configuration
        symbol = config['symbol']
        exchange = config['exchange']
        timeframe = config['timeframe']
        data_dir = config.get('data_dir', 'data')
        
        logger.info(f"Collecting data for {symbol} on {exchange} ({timeframe})")
        
        # Simulate data collection (replace with actual data collection logic)
        # In a real implementation, this would use the data downloader from utility_container
        data = await _simulate_data_collection(symbol, exchange, timeframe, config)
        
        # Validate collected data using unified data quality
        data_validation = validate_data_quality(data, 'ohlcv', 'comprehensive')
        
        if not data_validation['passed']:
            logger.warning(f"Data quality issues detected: {data_validation['errors']}")
            
            # Clean data if quality issues are found
            cleaned_data, cleaning_report = clean_data(data, 'standard')
            logger.info(f"Data cleaned: {cleaning_report['operations_performed']}")
            data = cleaned_data
        
        # Generate quality report
        quality_report = generate_quality_report(data, 'ohlcv')
        
        # Save data if configured
        if config.get('save_data', True):
            await _save_collected_data(data, data_dir, symbol, exchange, timeframe)
        
        return {
            'data': data,
            'data_validation': data_validation,
            'quality_report': quality_report,
            'collection_metadata': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'collected_at': datetime.now().isoformat(),
                'data_shape': data.shape if hasattr(data, 'shape') else None
            }
        }
        
    except Exception as e:
        logger.exception(f"Error in data collection logic: {e}")
        raise


async def _simulate_data_collection(symbol: str, exchange: str, timeframe: str, config: Dict[str, Any]):
    """
    Simulate data collection for demonstration purposes.
    
    In a real implementation, this would use the actual data downloader.
    """
    import pandas as pd
    import numpy as np
    
    logger.info(f"Simulating data collection for {symbol} on {exchange} ({timeframe})")
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    periods = config.get('periods', 1000)
    
    # Generate price data with some trend and volatility
    base_price = 100.0
    returns = np.random.normal(0, 0.02, periods)  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=periods, freq=timeframe),
        'open': prices * (1 + np.random.normal(0, 0.001, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, periods))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Add some realistic gaps and missing data
    if config.get('add_realistic_issues', True):
        # Add some missing values
        missing_indices = np.random.choice(len(data), size=int(0.01 * len(data)), replace=False)
        data.iloc[missing_indices, 1] = np.nan  # Missing open prices
        
        # Add some duplicates
        duplicate_indices = np.random.choice(len(data), size=int(0.005 * len(data)), replace=False)
        for idx in duplicate_indices:
            if idx > 0:
                data.iloc[idx] = data.iloc[idx-1]
    
    logger.info(f"Generated {len(data)} rows of OHLCV data")
    return data


async def _save_collected_data(data, data_dir: str, symbol: str, exchange: str, timeframe: str):
    """Save collected data to disk."""
    try:
        from src.utils.parquet_utils import get_parquet_utils
        
        # Ensure data directory exists
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{exchange}_{timeframe}_{timestamp}.parquet"
        filepath = Path(data_dir) / filename
        
        # Save using parquet utils
        parquet_utils = get_parquet_utils()
        parquet_utils.save_dataframe(data, str(filepath))
        
        logger.info(f"Data saved to: {filepath}")
        
    except Exception as e:
        logger.warning(f"Error saving data: {e}")


# Create simplified step function
step1_data_collection = create_simple_step_function("data_collection", step1_data_collection_logic)


class SimplifiedStep1DataCollection:
    """
    Simplified Step 1 Data Collection using new infrastructure.
    
    This replaces the complex Step1DataCollection class with a simple,
    utility-based approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified data collection."""
        self.config = validate_and_fix_config(config, 'data_collection')
        self.logger = logger.getChild('SimplifiedStep1DataCollection')
        
        # Initialize pipeline manager
        self.pipeline_manager = SimplifiedPipelineManager(self.config)
        
        # Add data collection step
        self.pipeline_manager.add_step("data_collection", step1_data_collection)
        
        self.logger.info("🚀 Simplified Step 1 Data Collection initialized")
    
    async def collect_data(self) -> Dict[str, Any]:
        """
        Collect data using simplified pipeline.
        
        Returns:
            Data collection result
        """
        try:
            self.logger.info("🚀 Starting data collection pipeline...")
            
            # Execute pipeline
            result = await self.pipeline_manager.execute_pipeline()
            
            if result['status'] == 'completed':
                self.logger.info("✅ Data collection completed successfully")
            else:
                self.logger.error(f"❌ Data collection failed: {result.get('errors', [])}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Data collection error: {e}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data."""
        try:
            pipeline_summary = self.pipeline_manager.get_pipeline_summary()
            
            # Extract data information from pipeline results
            step_results = pipeline_summary.get('step_results', {})
            data_collection_result = step_results.get('data_collection', {})
            
            summary = {
                'config': self.config,
                'pipeline_status': pipeline_summary.get('orchestrator_status', {}),
                'data_collection_result': data_collection_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add data information if available
            if 'data' in data_collection_result:
                data = data_collection_result['data']
                if hasattr(data, 'shape'):
                    summary['data_shape'] = data.shape
                if hasattr(data, 'columns'):
                    summary['data_columns'] = list(data.columns)
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"Error getting data summary: {e}")
            return {'error': str(e)}


# Backward compatibility wrapper
class Step1DataCollection(SimplifiedStep1DataCollection):
    """
    Backward compatibility wrapper for the original Step1DataCollection class.
    
    This allows existing code to continue using the old class name while
    benefiting from the new simplified infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with backward compatibility."""
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for Step1DataCollection")


# Example usage and testing
async def example_data_collection():
    """Example of using the simplified data collection."""
    
    # Configuration
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1m',
        'data_dir': 'data',
        'periods': 1000,
        'add_realistic_issues': True,
        'save_data': True
    }
    
    # Create simplified data collection
    collector = SimplifiedStep1DataCollection(config)
    
    # Collect data
    result = await collector.collect_data()
    
    # Get summary
    summary = collector.get_data_summary()
    
    print("=== Data Collection Result ===")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Data shape: {summary.get('data_shape', 'unknown')}")
    print(f"Data columns: {summary.get('data_columns', [])}")
    
    return result, summary


# Main execution
async def main():
    """Main execution function."""
    try:
        result, summary = await example_data_collection()
        print("✅ Data collection example completed successfully")
        return result, summary
    except Exception as e:
        logger.exception(f"Data collection example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())