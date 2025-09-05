#!/usr/bin/env python3
"""Data Collection Package for Trading Pipeline.

This package contains all the components for data collection:
- Raw data collection from exchanges
- Data quality validation and checking
- Unified data loading and preprocessing
- Data conversion and format standardization
- Integrated data quality pipeline
"""


# Main pipeline function
async def run_data_collection_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete data collection pipeline with enhanced protection."""
    try:
        # Import enhanced pipeline
        from .enhanced_data_collection_pipeline import run_enhanced_data_collection_pipeline
        
        # Run enhanced pipeline
        result = await run_enhanced_data_collection_pipeline(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
            config=config
        )
        
        return result.get("success", False)
        
    except Exception as e:
        print(f"Data collection pipeline failed: {e}")
        return False

# Import comprehensive monitoring modules
from .step01_enhanced_with_monitoring import (
    EnhancedDataCollectionStepWithMonitoring,
    run_enhanced_step01_with_monitoring
)

from .step01_comprehensive_monitoring import (
    Step01ComprehensiveMonitoring,
    run_comprehensive_step01
)

__all__ = [
    # Original modules
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
    'run_data_collection_pipeline',
    
    # Enhanced monitoring modules
    'EnhancedDataCollectionStepWithMonitoring',
    'run_enhanced_step01_with_monitoring',
    
    # Comprehensive monitoring modules
    'Step01ComprehensiveMonitoring',
    'run_comprehensive_step01'
]