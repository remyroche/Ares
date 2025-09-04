#!/usr/bin/env python3
"""Data Collection Package for Trading Pipeline.

This package contains all the components for data collection:
- Raw data collection from exchanges
- Data quality validation and checking
- Unified data loading and preprocessing
- Data conversion and format standardization
- Integrated data quality pipeline
"""

from .step01_data_collection import DataCollectionStep
from .step01_data_collection_validator import DataCollectionValidator
from .step01_5_data_converter_validator import DataConverterValidator
from .step02_data_reading import DataReadingStep
from .step02_data_reading_validator import DataReadingValidator
from .step02_5_sr_optimization_validator import SROptimizationValidator
from .unified_data_loader import UnifiedDataLoader
from .raw_data_quality_checker import RawDataQualityChecker
from .raw_data_quality_checker_refactored import RefactoredDataQualityChecker
from .integrated_data_quality_pipeline import IntegratedDataQualityPipeline

# Main pipeline function
async def run_data_collection_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete data collection pipeline."""
    try:
        # Step 1: Data Collection
        data_collector = DataCollectionStep()
        await data_collector.collect_data(symbol, exchange, timeframe, data_dir)
        
        # Step 2: Data Reading and Validation
        data_reader = DataReadingStep()
        data = await data_reader.read_data(symbol, exchange, timeframe, data_dir)
        
        # Step 3: Quality Validation
        quality_checker = RawDataQualityChecker()
        quality_results = await quality_checker.validate_data_quality(data)
        
        # Step 4: Unified Data Loading
        unified_loader = UnifiedDataLoader()
        processed_data = await unified_loader.load_and_process(data)
        
        return True
        
    except Exception as e:
        print(f"Data collection pipeline failed: {e}")
        return False

__all__ = [
    'DataCollectionStep',
    'DataCollectionValidator',
    'DataConverterValidator',
    'DataReadingStep',
    'DataReadingValidator',
    'SROptimizationValidator',
    'UnifiedDataLoader',
    'RawDataQualityChecker',
    'RefactoredDataQualityChecker',
    'IntegratedDataQualityPipeline',
    'run_data_collection_pipeline'
]