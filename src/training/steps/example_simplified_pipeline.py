"""
Example Simplified Pipeline

This file demonstrates the complete simplified pipeline infrastructure that replaces
the complex step-based approach with a unified, utility-based system.

Key Features:
- Uses SimplifiedPipelineManager for execution and monitoring
- Uses ConfigurationValidator for standardized config validation
- Uses DataQualityUtilities for unified data validation
- Uses MLPipelineOrchestrator for pipeline orchestration
- Simple function-based steps instead of complex classes
- Automatic error handling and recovery
- Comprehensive logging and monitoring
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# Import new simplified infrastructure
from .simplified_pipeline_infrastructure import (
    SimplifiedPipelineManager,
    create_simple_step_function,
    create_data_processing_step_function
)

# Import unified components
from .unified_feature_engineering import (
    unified_feature_engineering,
    comprehensive_feature_engineering
)

from .unified_feature_selection import (
    unified_feature_selection,
    comprehensive_feature_selection
)

from .unified_model_training import (
    unified_model_training,
    comprehensive_model_training
)

from .unified_model_evaluation import (
    unified_model_evaluation,
    comprehensive_model_evaluation
)

from .unified_optimization import (
    unified_optimization,
    comprehensive_optimization
)

# Import simplified steps
from .simplified_step1_data_collection import step1_data_collection
from .simplified_step5_labeling import step5_labeling

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class ExampleSimplifiedPipeline:
    """
    Example of a complete simplified pipeline using the new infrastructure.
    
    This demonstrates how to build a complete ML pipeline using the simplified
    infrastructure instead of complex step classes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the example simplified pipeline."""
        self.config = config
        self.logger = logger.getChild('ExampleSimplifiedPipeline')
        
        # Initialize pipeline manager
        self.pipeline_manager = SimplifiedPipelineManager(config)
        
        # Setup pipeline steps
        self._setup_pipeline()
        
        self.logger.info("🚀 Example Simplified Pipeline initialized")
    
    def _setup_pipeline(self):
        """Setup the complete pipeline with all steps."""
        try:
            self.logger.info("🔧 Setting up simplified pipeline...")
            
            # Step 1: Data Collection
            self.pipeline_manager.add_step("data_collection", step1_data_collection)
            
            # Step 2: Labeling (depends on data collection)
            self.pipeline_manager.add_step("labeling", step5_labeling, 
                                         dependencies=["data_collection"])
            
            # Step 3: Feature Engineering (depends on labeling)
            self.pipeline_manager.add_step("feature_engineering", comprehensive_feature_engineering,
                                         dependencies=["labeling"])
            
            # Step 4: Feature Selection (depends on feature engineering)
            self.pipeline_manager.add_step("feature_selection", comprehensive_feature_selection,
                                         dependencies=["feature_engineering"])
            
            # Step 5: Model Training (depends on feature selection)
            self.pipeline_manager.add_step("model_training", comprehensive_model_training,
                                         dependencies=["feature_selection"])
            
            # Step 6: Model Evaluation (depends on model training)
            self.pipeline_manager.add_step("model_evaluation", comprehensive_model_evaluation,
                                         dependencies=["model_training"])
            
            # Step 7: Optimization (depends on model evaluation)
            self.pipeline_manager.add_step("optimization", comprehensive_optimization,
                                         dependencies=["model_evaluation"])
            
            self.logger.info("✅ Pipeline setup completed with 7 steps")
            
        except Exception as e:
            self.logger.exception(f"Error setting up pipeline: {e}")
            raise
    
    async def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the complete simplified pipeline."""
        try:
            self.logger.info("🚀 Starting complete simplified pipeline execution...")
            
            # Execute pipeline
            result = await self.pipeline_manager.execute_pipeline()
            
            if result['status'] == 'completed':
                self.logger.info("✅ Complete simplified pipeline executed successfully")
            else:
                self.logger.error(f"❌ Pipeline execution failed: {result.get('errors', [])}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Pipeline execution error: {e}")
            raise
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary."""
        try:
            pipeline_summary = self.pipeline_manager.get_pipeline_summary()
            
            # Extract step results
            step_results = pipeline_summary.get('step_results', {})
            
            # Create comprehensive summary
            summary = {
                'config': self.config,
                'pipeline_status': pipeline_summary.get('orchestrator_status', {}),
                'step_results': step_results,
                'timestamp': datetime.now().isoformat(),
                'pipeline_info': self._get_pipeline_info()
            }
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"Error getting pipeline summary: {e}")
            return {'error': str(e)}
    
    def _get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            'pipeline_type': 'simplified_unified',
            'total_steps': 7,
            'steps': [
                'data_collection',
                'labeling',
                'feature_engineering',
                'feature_selection',
                'model_training',
                'model_evaluation',
                'optimization'
            ],
            'infrastructure_used': [
                'SimplifiedPipelineManager',
                'ConfigurationValidator',
                'DataQualityUtilities',
                'MLPipelineOrchestrator',
                'EnhancedFeatureEngineering',
                'Step08AdvancedFeatureSelection',
                'EnhancedModelTrainer',
                'ModelEvaluationUtilities',
                'MemoryEfficientTraining',
                'ParallelProcessingCoordinator'
            ],
            'benefits': [
                'Single unified approach',
                'Automatic validation and error handling',
                'Comprehensive monitoring and logging',
                'Built-in performance optimization',
                'Standardized configuration management',
                'Easy to maintain and extend'
            ]
        }


# Custom step functions for demonstration
async def custom_data_preprocessing_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Custom data preprocessing step logic."""
    logger.info("🔧 Executing custom data preprocessing...")
    
    try:
        # Get data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            raise ValueError("No data found in pipeline state")
        
        # Custom preprocessing logic
        processed_data = data.copy()
        
        # Example: Add custom features
        processed_data['custom_feature_1'] = processed_data['close'] / processed_data['volume']
        processed_data['custom_feature_2'] = processed_data['high'] - processed_data['low']
        
        # Example: Custom validation
        if processed_data.isnull().sum().sum() > 0:
            logger.warning("Missing values detected in processed data")
        
        return {
            'data': processed_data,
            'preprocessing_metadata': {
                'original_shape': data.shape,
                'processed_shape': processed_data.shape,
                'custom_features_added': 2,
                'missing_values': processed_data.isnull().sum().sum()
            }
        }
        
    except Exception as e:
        logger.exception(f"Error in custom data preprocessing: {e}")
        raise


async def custom_model_validation_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Custom model validation step logic."""
    logger.info("🔍 Executing custom model validation...")
    
    try:
        # Get model and evaluation results from pipeline state
        model = pipeline_state.get('model')
        evaluation_results = pipeline_state.get('model_evaluation', {})
        
        if model is None:
            raise ValueError("No model found in pipeline state")
        
        # Custom validation logic
        validation_result = {
            'model_valid': True,
            'validation_checks': [],
            'recommendations': []
        }
        
        # Example: Check model performance
        if 'evaluation_metrics' in evaluation_results:
            metrics = evaluation_results['evaluation_metrics']
            accuracy = metrics.get('accuracy', 0)
            
            if accuracy < 0.6:
                validation_result['model_valid'] = False
                validation_result['validation_checks'].append(f"Low accuracy: {accuracy:.3f}")
                validation_result['recommendations'].append("Consider feature engineering or model tuning")
            else:
                validation_result['validation_checks'].append(f"Good accuracy: {accuracy:.3f}")
        
        # Example: Check feature importance
        if 'feature_importance' in evaluation_results:
            importance = evaluation_results['feature_importance']
            if isinstance(importance, dict):
                max_importance = max(importance.values()) if importance else 0
                if max_importance < 0.1:
                    validation_result['recommendations'].append("Consider feature selection to improve model interpretability")
        
        return {
            'validation_result': validation_result,
            'model_approved': validation_result['model_valid']
        }
        
    except Exception as e:
        logger.exception(f"Error in custom model validation: {e}")
        raise


# Create custom step functions
custom_data_preprocessing = create_data_processing_step_function("custom_data_preprocessing", custom_data_preprocessing_logic)
custom_model_validation = create_simple_step_function("custom_model_validation", custom_model_validation_logic)


class CustomSimplifiedPipeline(ExampleSimplifiedPipeline):
    """
    Custom simplified pipeline with additional steps.
    
    This demonstrates how to extend the simplified pipeline with custom steps.
    """
    
    def _setup_pipeline(self):
        """Setup the custom pipeline with additional steps."""
        try:
            self.logger.info("🔧 Setting up custom simplified pipeline...")
            
            # Standard steps
            self.pipeline_manager.add_step("data_collection", step1_data_collection)
            self.pipeline_manager.add_step("labeling", step5_labeling, 
                                         dependencies=["data_collection"])
            
            # Custom preprocessing step
            self.pipeline_manager.add_step("custom_preprocessing", custom_data_preprocessing,
                                         dependencies=["labeling"])
            
            # Feature engineering (depends on custom preprocessing)
            self.pipeline_manager.add_step("feature_engineering", comprehensive_feature_engineering,
                                         dependencies=["custom_preprocessing"])
            
            # Feature selection
            self.pipeline_manager.add_step("feature_selection", comprehensive_feature_selection,
                                         dependencies=["feature_engineering"])
            
            # Model training
            self.pipeline_manager.add_step("model_training", comprehensive_model_training,
                                         dependencies=["feature_selection"])
            
            # Model evaluation
            self.pipeline_manager.add_step("model_evaluation", comprehensive_model_evaluation,
                                         dependencies=["model_training"])
            
            # Custom model validation
            self.pipeline_manager.add_step("custom_validation", custom_model_validation,
                                         dependencies=["model_evaluation"])
            
            # Optimization (depends on custom validation)
            self.pipeline_manager.add_step("optimization", comprehensive_optimization,
                                         dependencies=["custom_validation"])
            
            self.logger.info("✅ Custom pipeline setup completed with 9 steps")
            
        except Exception as e:
            self.logger.exception(f"Error setting up custom pipeline: {e}")
            raise


async def demonstrate_simplified_pipeline():
    """
    Demonstrate the simplified pipeline infrastructure.
    """
    logger.info("🚀 Demonstrating Simplified Pipeline Infrastructure")
    
    # Configuration
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1m',
        'data_dir': 'data',
        'output_dir': 'output',
        'model_dir': 'models',
        'log_dir': 'logs',
        'enable_gpu': True,
        'enable_parallel': True,
        'max_workers': 4,
        'memory_limit': 0.8,
        'timeout_seconds': 3600,
        'random_state': 42,
        
        # Feature engineering configuration
        'feature_engineering_config': {
            'enable_technical_indicators': True,
            'enable_statistical_features': True,
            'enable_lag_features': True,
            'enable_interaction_features': True,
            'enable_regime_features': True,
            'enable_wavelet_features': True,
            'enable_multi_timeframe_features': True,
            'max_lags': 10,
            'max_interactions': 20,
            'max_features': 100
        },
        
        # Feature selection configuration
        'feature_selection_config': {
            'selection_method': 'mrmr',
            'n_features': 50,
            'stability_threshold': 0.6,
            'enable_regime_specific': False
        },
        
        # Model training configuration
        'model_training_config': {
            'enable_confidence_metrics': True,
            'enable_calibration_assessment': True,
            'enable_feature_importance': True,
            'enable_cross_validation': True,
            'enable_model_explanations': True,
            'enable_post_training_hpo': True,
            'cv_folds': 5
        },
        
        # Model evaluation configuration
        'evaluation_config': {
            'enable_cross_validation': True,
            'enable_time_series_validation': True,
            'enable_confidence_intervals': True,
            'enable_model_comparison': True,
            'enable_feature_importance_analysis': True,
            'enable_prediction_analysis': True,
            'cv_folds': 5,
            'confidence_level': 0.95
        },
        
        # Optimization configuration
        'optimization_config': {
            'enable_memory_optimization': True,
            'enable_parallel_processing': True,
            'enable_m1_optimizations': True,
            'enable_gpu_acceleration': True,
            'enable_automatic_chunking': True,
            'enable_memory_monitoring': True,
            'enable_performance_profiling': True,
            'chunk_size_mb': 200,
            'max_workers': 4
        }
    }
    
    print("=" * 80)
    print("SIMPLIFIED PIPELINE INFRASTRUCTURE DEMONSTRATION")
    print("=" * 80)
    
    # Standard simplified pipeline
    print("\n🚀 Standard Simplified Pipeline")
    print("-" * 50)
    
    standard_pipeline = ExampleSimplifiedPipeline(config)
    standard_result = await standard_pipeline.execute_pipeline()
    standard_summary = standard_pipeline.get_pipeline_summary()
    
    print(f"✅ Pipeline status: {standard_result.get('status', 'unknown')}")
    print(f"📊 Total steps: {len(standard_summary['pipeline_info']['steps'])}")
    print(f"🔧 Infrastructure used: {len(standard_summary['pipeline_info']['infrastructure_used'])} components")
    print(f"📁 Benefits: {len(standard_summary['pipeline_info']['benefits'])} key improvements")
    
    # Custom simplified pipeline
    print("\n🎯 Custom Simplified Pipeline")
    print("-" * 50)
    
    custom_pipeline = CustomSimplifiedPipeline(config)
    custom_result = await custom_pipeline.execute_pipeline()
    custom_summary = custom_pipeline.get_pipeline_summary()
    
    print(f"✅ Pipeline status: {custom_result.get('status', 'unknown')}")
    print(f"📊 Total steps: {len(custom_summary['pipeline_info']['steps'])}")
    print(f"🔧 Custom steps added: 2 (preprocessing, validation)")
    print(f"📁 Pipeline type: {custom_summary['pipeline_info']['pipeline_type']}")
    
    # Infrastructure benefits
    print("\n📊 INFRASTRUCTURE BENEFITS")
    print("=" * 50)
    
    benefits = standard_summary['pipeline_info']['benefits']
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
    
    # Infrastructure components
    print("\n🔧 INFRASTRUCTURE COMPONENTS")
    print("=" * 50)
    
    components = standard_summary['pipeline_info']['infrastructure_used']
    for i, component in enumerate(components, 1):
        print(f"{i}. {component}")
    
    return {
        'standard_result': standard_result,
        'standard_summary': standard_summary,
        'custom_result': custom_result,
        'custom_summary': custom_summary,
        'infrastructure_benefits': benefits,
        'infrastructure_components': components
    }


# Main execution
async def main():
    """Main execution function."""
    try:
        results = await demonstrate_simplified_pipeline()
        print("\n✅ Simplified Pipeline demonstration completed successfully")
        return results
    except Exception as e:
        logger.exception(f"Simplified Pipeline demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())