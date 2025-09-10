"""
Example Simplified Pipeline

This module demonstrates how to use the new simplified infrastructure for training steps
using MLPipelineOrchestrator and utility-based approaches.

Key Features:
- Complete pipeline example using SimplifiedPipelineManager
- Demonstrates data collection, labeling, and feature engineering
- Shows how to use standardized configuration validation
- Demonstrates unified data quality management
- Shows error handling and recovery
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

# Import simplified steps
from .simplified_step1_data_collection import step1_data_collection
from .simplified_step5_labeling import step5_labeling

# Import step06 utilities for feature engineering
from src.utils.step06_utilities import (
    EnhancedFeatureEngineering,
    get_utility_container
)

# Import ML Common utilities
from src.utils.ml_common import (
    FeatureSelectionFramework,
    EnhancedModelTrainer
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


# Additional step functions for the example pipeline
async def step2_feature_engineering_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Feature engineering step using step06 utilities."""
    logger.info("🔧 Starting feature engineering...")
    
    try:
        # Get data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            raise ValueError("No data found in pipeline state for feature engineering")
        
        # Get utility container
        utility_container = get_utility_container(config)
        
        # Initialize enhanced feature engineering
        feature_engine = EnhancedFeatureEngineering(config)
        
        # Create advanced features
        features = feature_engine.create_advanced_features(
            data=data,
            enable_gpu_acceleration=config.get('enable_gpu', True),
            enable_parallel_processing=config.get('enable_parallel', True)
        )
        
        # Validate features
        features_validation = validate_data_quality(features, 'features', 'comprehensive')
        
        return {
            'features': features,
            'feature_metadata': feature_engine.get_feature_metadata(),
            'features_validation': features_validation
        }
        
    except Exception as e:
        logger.exception(f"Error in feature engineering: {e}")
        raise


async def step3_feature_selection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Feature selection step using ML Common utilities."""
    logger.info("🎯 Starting feature selection...")
    
    try:
        # Get features and labels from pipeline state
        features = pipeline_state.get('features')
        labels = pipeline_state.get('labels')
        
        if features is None or labels is None:
            raise ValueError("Missing features or labels in pipeline state for feature selection")
        
        # Initialize feature selection framework
        feature_selector = FeatureSelectionFramework(config.get('feature_selection_config', {}))
        
        # Perform feature selection
        selection_result = feature_selector.select_features(
            X=features,
            y=labels,
            method=config.get('selection_method', 'mrmr'),
            n_features=config.get('n_features', 50)
        )
        
        # Validate selected features
        selected_features_validation = validate_data_quality(
            selection_result['selected_features'], 'features', 'standard'
        )
        
        return {
            'selected_features': selection_result['selected_features'],
            'feature_importance': selection_result.get('feature_importance', {}),
            'selection_metadata': selection_result.get('metadata', {}),
            'selected_features_validation': selected_features_validation
        }
        
    except Exception as e:
        logger.exception(f"Error in feature selection: {e}")
        raise


async def step4_model_training_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Model training step using ML Common utilities."""
    logger.info("🤖 Starting model training...")
    
    try:
        # Get selected features and labels from pipeline state
        selected_features = pipeline_state.get('selected_features')
        labels = pipeline_state.get('labels')
        
        if selected_features is None or labels is None:
            raise ValueError("Missing selected features or labels in pipeline state for model training")
        
        # Initialize enhanced model trainer
        trainer = EnhancedModelTrainer(config.get('model_training_config', {}))
        
        # Split data for training and testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            selected_features, labels, 
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42),
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Train and evaluate model
        training_result = trainer.train_and_evaluate_model(
            model=config.get('model_class', 'RandomForestClassifier')(),
            model_name="example_model",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=list(selected_features.columns) if hasattr(selected_features, 'columns') else None
        )
        
        return {
            'model': training_result['model'],
            'evaluation_metrics': training_result['evaluation_metrics'],
            'confidence_metrics': training_result.get('confidence_metrics', {}),
            'feature_importance': training_result.get('feature_importance', {}),
            'training_metadata': training_result.get('metadata', {})
        }
        
    except Exception as e:
        logger.exception(f"Error in model training: {e}")
        raise


# Create step functions
step2_feature_engineering = create_data_processing_step_function("feature_engineering", step2_feature_engineering_logic)
step3_feature_selection = create_simple_step_function("feature_selection", step3_feature_selection_logic)
step4_model_training = create_simple_step_function("model_training", step4_model_training_logic)


class ExampleSimplifiedPipeline:
    """
    Example simplified pipeline demonstrating the new infrastructure.
    
    This shows how to create a complete ML pipeline using the simplified
    infrastructure with MLPipelineOrchestrator and utility-based approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize example pipeline."""
        self.config = validate_and_fix_config(config)
        self.logger = logger.getChild('ExampleSimplifiedPipeline')
        
        # Initialize pipeline manager
        self.pipeline_manager = SimplifiedPipelineManager(self.config)
        
        # Add all steps to pipeline
        self._setup_pipeline()
        
        self.logger.info("🚀 Example Simplified Pipeline initialized")
    
    def _setup_pipeline(self):
        """Setup the complete pipeline with all steps."""
        try:
            # Step 1: Data Collection
            self.pipeline_manager.add_step("data_collection", step1_data_collection)
            
            # Step 2: Feature Engineering (depends on data_collection)
            self.pipeline_manager.add_step(
                "feature_engineering", 
                step2_feature_engineering,
                dependencies=["data_collection"]
            )
            
            # Step 3: Labeling (depends on data_collection)
            self.pipeline_manager.add_step(
                "labeling",
                step5_labeling,
                dependencies=["data_collection"]
            )
            
            # Step 4: Feature Selection (depends on feature_engineering and labeling)
            self.pipeline_manager.add_step(
                "feature_selection",
                step3_feature_selection,
                dependencies=["feature_engineering", "labeling"]
            )
            
            # Step 5: Model Training (depends on feature_selection and labeling)
            self.pipeline_manager.add_step(
                "model_training",
                step4_model_training,
                dependencies=["feature_selection", "labeling"]
            )
            
            self.logger.info("✅ Pipeline setup completed with 5 steps")
            
        except Exception as e:
            self.logger.exception(f"Error setting up pipeline: {e}")
            raise
    
    async def execute_pipeline(self) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        try:
            self.logger.info("🚀 Starting complete pipeline execution...")
            
            # Execute pipeline
            result = await self.pipeline_manager.execute_pipeline()
            
            if result['status'] == 'completed':
                self.logger.info("✅ Complete pipeline execution completed successfully")
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
                'pipeline_metrics': self._calculate_pipeline_metrics(step_results)
            }
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"Error getting pipeline summary: {e}")
            return {'error': str(e)}
    
    def _calculate_pipeline_metrics(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate pipeline performance metrics."""
        try:
            metrics = {
                'total_steps': len(step_results),
                'completed_steps': 0,
                'failed_steps': 0,
                'data_quality_scores': {},
                'model_performance': {}
            }
            
            for step_name, step_result in step_results.items():
                if step_result.get('status') == 'completed':
                    metrics['completed_steps'] += 1
                else:
                    metrics['failed_steps'] += 1
                
                # Extract data quality scores
                if 'data_validation' in step_result:
                    validation = step_result['data_validation']
                    if 'quality_score' in validation:
                        metrics['data_quality_scores'][step_name] = validation['quality_score']
                
                # Extract model performance
                if step_name == 'model_training' and 'evaluation_metrics' in step_result:
                    metrics['model_performance'] = step_result['evaluation_metrics']
            
            metrics['success_rate'] = metrics['completed_steps'] / metrics['total_steps'] if metrics['total_steps'] > 0 else 0
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error calculating pipeline metrics: {e}")
            return {'error': str(e)}


# Example usage and testing
async def example_complete_pipeline():
    """Example of using the complete simplified pipeline."""
    
    # Comprehensive configuration
    config = {
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'timeframe': '1m',
        'data_dir': 'data',
        'output_dir': 'output',
        'model_dir': 'models',
        
        # Data collection config
        'periods': 2000,
        'add_realistic_issues': True,
        'save_data': True,
        
        # Labeling config
        'labeling_config': {
            'method': 'triple_barrier',
            'upper_threshold': 0.02,
            'lower_threshold': -0.02,
            'max_holding_period': 20
        },
        
        # Feature engineering config
        'feature_engineering_config': {
            'enable_technical_indicators': True,
            'enable_statistical_features': True,
            'enable_lag_features': True,
            'max_lags': 10
        },
        
        # Feature selection config
        'feature_selection_config': {
            'method': 'mrmr',
            'n_features': 50,
            'stability_threshold': 0.6
        },
        'selection_method': 'mrmr',
        'n_features': 50,
        
        # Model training config
        'model_training_config': {
            'enable_confidence_metrics': True,
            'enable_calibration_assessment': True,
            'enable_feature_importance': True,
            'cv_folds': 5
        },
        'model_class': 'RandomForestClassifier',
        'test_size': 0.2,
        'random_state': 42,
        
        # Performance config
        'enable_gpu': True,
        'enable_parallel': True,
        'max_workers': 4,
        'memory_limit': 0.8,
        'timeout_seconds': 3600
    }
    
    print("=== Example Simplified Pipeline ===")
    print(f"Configuration: {config['symbol']} on {config['exchange']} ({config['timeframe']})")
    
    # Create and execute pipeline
    pipeline = ExampleSimplifiedPipeline(config)
    
    # Execute complete pipeline
    result = await pipeline.execute_pipeline()
    
    # Get comprehensive summary
    summary = pipeline.get_pipeline_summary()
    
    # Display results
    print(f"\n=== Pipeline Execution Results ===")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Success Rate: {summary.get('pipeline_metrics', {}).get('success_rate', 0):.2%}")
    print(f"Completed Steps: {summary.get('pipeline_metrics', {}).get('completed_steps', 0)}")
    print(f"Failed Steps: {summary.get('pipeline_metrics', {}).get('failed_steps', 0)}")
    
    # Display data quality scores
    quality_scores = summary.get('pipeline_metrics', {}).get('data_quality_scores', {})
    if quality_scores:
        print(f"\n=== Data Quality Scores ===")
        for step, score in quality_scores.items():
            print(f"{step}: {score:.3f}")
    
    # Display model performance
    model_performance = summary.get('pipeline_metrics', {}).get('model_performance', {})
    if model_performance:
        print(f"\n=== Model Performance ===")
        for metric, value in model_performance.items():
            print(f"{metric}: {value:.3f}")
    
    return result, summary


# Main execution
async def main():
    """Main execution function."""
    try:
        result, summary = await example_complete_pipeline()
        print("\n✅ Example pipeline completed successfully")
        return result, summary
    except Exception as e:
        logger.exception(f"Example pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())