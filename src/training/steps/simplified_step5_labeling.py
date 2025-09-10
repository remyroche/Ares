"""
Simplified Step 5: Labeling

This module provides a simplified version of step5_labeling using the new
infrastructure with MLPipelineOrchestrator and utility-based approaches.

Key Features:
- Uses SimplifiedPipelineManager for execution and monitoring
- Uses ConfigurationValidator for standardized config validation
- Uses DataQualityUtilities for unified data validation
- Uses step06_utilities for labeling components
- Simple function-based approach instead of complex class
- Automatic error handling and recovery
- Comprehensive logging and monitoring
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

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

# Import step06 utilities for labeling components
from src.utils.step06_utilities import (
    Step06UtilityContainer,
    get_utility_container,
    OptimizedTripleBarrierLabeling,
    FractionalTripleBarrierLabeling,
    RegimeSpecificTripleBarrierOptimizer
)

# Import ML Common utilities
from src.utils.ml_common import (
    DataQualityUtilities,
    MLTrainingSafeguards
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


# Simplified labeling step function
async def step5_labeling_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified labeling logic using utilities.
    
    Args:
        config: Configuration dictionary
        pipeline_state: Current pipeline state
        
    Returns:
        Labeling result
    """
    logger.info("🏷️ Starting simplified labeling...")
    
    try:
        # Get data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            raise ValueError("No data found in pipeline state for labeling")
        
        # Validate input data
        data_validation = validate_data_quality(data, 'ohlcv', 'comprehensive')
        if not data_validation['passed']:
            logger.warning(f"Input data quality issues: {data_validation['errors']}")
        
        # Get utility container for dependency injection
        utility_container = get_utility_container(config)
        
        # Extract labeling configuration
        labeling_config = config.get('labeling_config', {})
        labeling_method = labeling_config.get('method', 'triple_barrier')
        
        logger.info(f"Using labeling method: {labeling_method}")
        
        # Perform labeling based on method
        if labeling_method == 'triple_barrier':
            labels, labeling_metadata = await _perform_triple_barrier_labeling(
                data, labeling_config, utility_container
            )
        elif labeling_method == 'fractional_triple_barrier':
            labels, labeling_metadata = await _perform_fractional_triple_barrier_labeling(
                data, labeling_config, utility_container
            )
        elif labeling_method == 'regime_specific':
            labels, labeling_metadata = await _perform_regime_specific_labeling(
                data, labeling_config, utility_container
            )
        else:
            raise ValueError(f"Unknown labeling method: {labeling_method}")
        
        # Validate labels
        labels_validation = validate_data_quality(labels, 'targets', 'standard')
        
        # Create labeled dataset
        labeled_data = data.copy()
        labeled_data['labels'] = labels
        
        # Generate quality report for labeled data
        quality_report = generate_quality_report(labeled_data, 'labeled_data')
        
        return {
            'labeled_data': labeled_data,
            'labels': labels,
            'labeling_metadata': labeling_metadata,
            'data_validation': data_validation,
            'labels_validation': labels_validation,
            'quality_report': quality_report,
            'labeling_config': labeling_config
        }
        
    except Exception as e:
        logger.exception(f"Error in labeling logic: {e}")
        raise


async def _perform_triple_barrier_labeling(data: pd.DataFrame, config: Dict[str, Any], 
                                         utility_container: Step06UtilityContainer) -> Tuple[pd.Series, Dict[str, Any]]:
    """Perform triple barrier labeling using step06 utilities."""
    try:
        logger.info("Performing triple barrier labeling...")
        
        # Initialize triple barrier labeling
        triple_barrier = OptimizedTripleBarrierLabeling(config)
        
        # Extract parameters
        upper_threshold = config.get('upper_threshold', 0.02)  # 2%
        lower_threshold = config.get('lower_threshold', -0.02)  # -2%
        max_holding_period = config.get('max_holding_period', 20)  # 20 periods
        
        # Perform labeling
        labels = triple_barrier.label_data(
            data=data,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            max_holding_period=max_holding_period
        )
        
        # Generate metadata
        metadata = {
            'method': 'triple_barrier',
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'max_holding_period': max_holding_period,
            'label_distribution': labels.value_counts().to_dict(),
            'label_ratio': labels.value_counts(normalize=True).to_dict()
        }
        
        logger.info(f"Triple barrier labeling completed. Label distribution: {metadata['label_distribution']}")
        
        return labels, metadata
        
    except Exception as e:
        logger.exception(f"Error in triple barrier labeling: {e}")
        raise


async def _perform_fractional_triple_barrier_labeling(data: pd.DataFrame, config: Dict[str, Any], 
                                                    utility_container: Step06UtilityContainer) -> Tuple[pd.Series, Dict[str, Any]]:
    """Perform fractional triple barrier labeling using step06 utilities."""
    try:
        logger.info("Performing fractional triple barrier labeling...")
        
        # Initialize fractional triple barrier labeling
        fractional_triple_barrier = FractionalTripleBarrierLabeling(config)
        
        # Extract parameters
        upper_threshold = config.get('upper_threshold', 0.02)
        lower_threshold = config.get('lower_threshold', -0.02)
        max_holding_period = config.get('max_holding_period', 20)
        fractional_threshold = config.get('fractional_threshold', 0.5)
        
        # Perform labeling
        labels = fractional_triple_barrier.label_data(
            data=data,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            max_holding_period=max_holding_period,
            fractional_threshold=fractional_threshold
        )
        
        # Generate metadata
        metadata = {
            'method': 'fractional_triple_barrier',
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'max_holding_period': max_holding_period,
            'fractional_threshold': fractional_threshold,
            'label_distribution': labels.value_counts().to_dict(),
            'label_ratio': labels.value_counts(normalize=True).to_dict()
        }
        
        logger.info(f"Fractional triple barrier labeling completed. Label distribution: {metadata['label_distribution']}")
        
        return labels, metadata
        
    except Exception as e:
        logger.exception(f"Error in fractional triple barrier labeling: {e}")
        raise


async def _perform_regime_specific_labeling(data: pd.DataFrame, config: Dict[str, Any], 
                                          utility_container: Step06UtilityContainer) -> Tuple[pd.Series, Dict[str, Any]]:
    """Perform regime-specific labeling using step06 utilities."""
    try:
        logger.info("Performing regime-specific labeling...")
        
        # Initialize regime-specific triple barrier optimizer
        regime_optimizer = RegimeSpecificTripleBarrierOptimizer(config)
        
        # Extract parameters
        regime_config = config.get('regime_config', {})
        default_thresholds = config.get('default_thresholds', {
            'upper_threshold': 0.02,
            'lower_threshold': -0.02,
            'max_holding_period': 20
        })
        
        # Perform labeling
        labels = regime_optimizer.optimize_and_label(
            data=data,
            regime_config=regime_config,
            default_thresholds=default_thresholds
        )
        
        # Generate metadata
        metadata = {
            'method': 'regime_specific',
            'regime_config': regime_config,
            'default_thresholds': default_thresholds,
            'label_distribution': labels.value_counts().to_dict(),
            'label_ratio': labels.value_counts(normalize=True).to_dict(),
            'regime_optimization_results': regime_optimizer.get_optimization_results()
        }
        
        logger.info(f"Regime-specific labeling completed. Label distribution: {metadata['label_distribution']}")
        
        return labels, metadata
        
    except Exception as e:
        logger.exception(f"Error in regime-specific labeling: {e}")
        raise


# Create simplified step function
step5_labeling = create_data_processing_step_function("labeling", step5_labeling_logic)


class SimplifiedStep5Labeling:
    """
    Simplified Step 5 Labeling using new infrastructure.
    
    This replaces the complex Step5Labeling class with a simple,
    utility-based approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified labeling."""
        self.config = validate_and_fix_config(config, 'labeling')
        self.logger = logger.getChild('SimplifiedStep5Labeling')
        
        # Initialize pipeline manager
        self.pipeline_manager = SimplifiedPipelineManager(self.config)
        
        # Add labeling step
        self.pipeline_manager.add_step("labeling", step5_labeling)
        
        self.logger.info("🚀 Simplified Step 5 Labeling initialized")
    
    async def label_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Label data using simplified pipeline.
        
        Args:
            data: Data to label
            
        Returns:
            Labeling result
        """
        try:
            self.logger.info("🚀 Starting labeling pipeline...")
            
            # Set data in pipeline state
            self.pipeline_manager.pipeline_state['data'] = data
            
            # Execute pipeline
            result = await self.pipeline_manager.execute_pipeline()
            
            if result['status'] == 'completed':
                self.logger.info("✅ Labeling completed successfully")
            else:
                self.logger.error(f"❌ Labeling failed: {result.get('errors', [])}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Labeling error: {e}")
            raise
    
    def get_labeling_summary(self) -> Dict[str, Any]:
        """Get summary of labeling results."""
        try:
            pipeline_summary = self.pipeline_manager.get_pipeline_summary()
            
            # Extract labeling information from pipeline results
            step_results = pipeline_summary.get('step_results', {})
            labeling_result = step_results.get('labeling', {})
            
            summary = {
                'config': self.config,
                'pipeline_status': pipeline_summary.get('orchestrator_status', {}),
                'labeling_result': labeling_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add labeling information if available
            if 'labeling_metadata' in labeling_result:
                metadata = labeling_result['labeling_metadata']
                summary['labeling_method'] = metadata.get('method', 'unknown')
                summary['label_distribution'] = metadata.get('label_distribution', {})
                summary['label_ratio'] = metadata.get('label_ratio', {})
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"Error getting labeling summary: {e}")
            return {'error': str(e)}


# Backward compatibility wrapper
class LabelingStep(SimplifiedStep5Labeling):
    """
    Backward compatibility wrapper for the original LabelingStep class.
    
    This allows existing code to continue using the old class name while
    benefiting from the new simplified infrastructure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with backward compatibility."""
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for LabelingStep")


# Example usage and testing
async def example_labeling():
    """Example of using the simplified labeling."""
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Configuration for different labeling methods
    configs = [
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'labeling_config': {
                'method': 'triple_barrier',
                'upper_threshold': 0.02,
                'lower_threshold': -0.02,
                'max_holding_period': 20
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'labeling_config': {
                'method': 'fractional_triple_barrier',
                'upper_threshold': 0.02,
                'lower_threshold': -0.02,
                'max_holding_period': 20,
                'fractional_threshold': 0.5
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Labeling Method {i+1}: {config['labeling_config']['method']} ===")
        
        # Create simplified labeling
        labeler = SimplifiedStep5Labeling(config)
        
        # Label data
        result = await labeler.label_data(data)
        
        # Get summary
        summary = labeler.get_labeling_summary()
        
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Labeling method: {summary.get('labeling_method', 'unknown')}")
        print(f"Label distribution: {summary.get('label_distribution', {})}")
        print(f"Label ratio: {summary.get('label_ratio', {})}")
        
        results.append((result, summary))
    
    return results


# Main execution
async def main():
    """Main execution function."""
    try:
        results = await example_labeling()
        print("✅ Labeling example completed successfully")
        return results
    except Exception as e:
        logger.exception(f"Labeling example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())