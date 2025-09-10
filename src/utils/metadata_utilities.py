"""
Metadata Generation Utilities

This module provides metadata generation utilities extracted from training steps
to eliminate code duplication and provide consistent metadata handling across all steps.

Key Features:
- Common metadata generation patterns
- Timestamp and execution tracking
- Result aggregation utilities
- Metadata validation and formatting
- Integration with pipeline infrastructure
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import hashlib

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class MetadataUtilities:
    """
    Metadata generation utilities for all training steps.
    
    This provides common metadata generation patterns and utilities
    extracted from multiple training step implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metadata utilities."""
        self.config = config or {}
        self.logger = logger.getChild('MetadataUtilities')
        
        self.logger.info("🚀 Metadata Utilities initialized")
    
    def generate_step_metadata(self, step_name: str, step_type: str, 
                             result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata for a step execution.
        
        Args:
            step_name: Name of the step
            step_type: Type of step (data_collection, feature_engineering, etc.)
            result: Step execution result
            config: Configuration used
            
        Returns:
            Step metadata dictionary
        """
        try:
            metadata = {
                'step_name': step_name,
                'step_type': step_type,
                'execution_timestamp': datetime.now().isoformat(),
                'status': result.get('status', 'unknown'),
                'config_hash': self._generate_config_hash(config),
                'result_summary': self._generate_result_summary(result),
                'execution_info': {
                    'step_name': step_name,
                    'step_type': step_type,
                    'executed_at': datetime.now().isoformat(),
                    'status': result.get('status', 'unknown')
                }
            }
            
            # Add step-specific metadata
            if step_type == 'data_collection':
                metadata.update(self._generate_data_collection_metadata(result))
            elif step_type == 'feature_engineering':
                metadata.update(self._generate_feature_engineering_metadata(result))
            elif step_type == 'model_training':
                metadata.update(self._generate_model_training_metadata(result))
            elif step_type == 'model_evaluation':
                metadata.update(self._generate_model_evaluation_metadata(result))
            elif step_type == 'optimization':
                metadata.update(self._generate_optimization_metadata(result))
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating step metadata: {e}")
            return {
                'step_name': step_name,
                'step_type': step_type,
                'execution_timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration for tracking changes."""
        try:
            # Sort config to ensure consistent hashing
            sorted_config = json.dumps(config, sort_keys=True, default=str)
            return hashlib.md5(sorted_config.encode()).hexdigest()[:8]
        except Exception as e:
            self.logger.warning(f"Error generating config hash: {e}")
            return "unknown"
    
    def _generate_result_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of step result."""
        try:
            summary = {
                'has_data': 'data' in result,
                'has_model': 'model' in result,
                'has_metrics': 'evaluation_metrics' in result,
                'has_validation': 'validation' in result or 'passed' in result,
                'result_keys': list(result.keys())
            }
            
            # Add data shape if present
            if 'data' in result:
                data = result['data']
                if hasattr(data, 'shape'):
                    summary['data_shape'] = data.shape
                elif isinstance(data, dict):
                    summary['data_keys'] = list(data.keys())
            
            # Add model info if present
            if 'model' in result:
                model = result['model']
                summary['model_type'] = type(model).__name__
                if hasattr(model, 'get_params'):
                    summary['model_params_count'] = len(model.get_params())
            
            # Add metrics summary if present
            if 'evaluation_metrics' in result:
                metrics = result['evaluation_metrics']
                summary['metrics_count'] = len(metrics)
                summary['metrics_keys'] = list(metrics.keys())
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Error generating result summary: {e}")
            return {'error': str(e)}
    
    def _generate_data_collection_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata specific to data collection steps."""
        try:
            metadata = {
                'data_collection_info': {
                    'data_collected': 'data' in result,
                    'collection_timestamp': datetime.now().isoformat()
                }
            }
            
            if 'data' in result:
                data = result['data']
                if hasattr(data, 'shape'):
                    metadata['data_collection_info'].update({
                        'data_shape': data.shape,
                        'data_size_mb': self._estimate_data_size(data)
                    })
                
                if hasattr(data, 'columns'):
                    metadata['data_collection_info']['columns'] = list(data.columns)
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating data collection metadata: {e}")
            return {}
    
    def _generate_feature_engineering_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata specific to feature engineering steps."""
        try:
            metadata = {
                'feature_engineering_info': {
                    'features_created': 'features' in result or 'data' in result,
                    'engineering_timestamp': datetime.now().isoformat()
                }
            }
            
            # Get features from result
            features = result.get('features') or result.get('data')
            if features is not None and hasattr(features, 'shape'):
                metadata['feature_engineering_info'].update({
                    'feature_count': features.shape[1],
                    'sample_count': features.shape[0],
                    'feature_names': list(features.columns) if hasattr(features, 'columns') else None
                })
            
            # Add feature metadata if present
            if 'feature_metadata' in result:
                feature_metadata = result['feature_metadata']
                metadata['feature_engineering_info']['feature_metadata'] = feature_metadata
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating feature engineering metadata: {e}")
            return {}
    
    def _generate_model_training_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata specific to model training steps."""
        try:
            metadata = {
                'model_training_info': {
                    'model_trained': 'model' in result,
                    'training_timestamp': datetime.now().isoformat()
                }
            }
            
            if 'model' in result:
                model = result['model']
                metadata['model_training_info'].update({
                    'model_type': type(model).__name__,
                    'model_params': model.get_params() if hasattr(model, 'get_params') else {}
                })
            
            # Add training metadata if present
            if 'training_metadata' in result:
                training_metadata = result['training_metadata']
                metadata['model_training_info']['training_metadata'] = training_metadata
            
            # Add evaluation metrics if present
            if 'evaluation_metrics' in result:
                eval_metrics = result['evaluation_metrics']
                metadata['model_training_info']['evaluation_metrics'] = eval_metrics
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating model training metadata: {e}")
            return {}
    
    def _generate_model_evaluation_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata specific to model evaluation steps."""
        try:
            metadata = {
                'model_evaluation_info': {
                    'evaluation_completed': 'evaluation_metrics' in result,
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            }
            
            if 'evaluation_metrics' in result:
                eval_metrics = result['evaluation_metrics']
                metadata['model_evaluation_info'].update({
                    'metrics_count': len(eval_metrics),
                    'metrics_keys': list(eval_metrics.keys()),
                    'primary_metric': eval_metrics.get('accuracy', eval_metrics.get('test_accuracy', 0))
                })
            
            # Add evaluation metadata if present
            if 'evaluation_metadata' in result:
                eval_metadata = result['evaluation_metadata']
                metadata['model_evaluation_info']['evaluation_metadata'] = eval_metadata
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating model evaluation metadata: {e}")
            return {}
    
    def _generate_optimization_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata specific to optimization steps."""
        try:
            metadata = {
                'optimization_info': {
                    'optimization_completed': 'optimization_result' in result,
                    'optimization_timestamp': datetime.now().isoformat()
                }
            }
            
            # Add performance metrics if present
            if 'performance_metrics' in result:
                perf_metrics = result['performance_metrics']
                metadata['optimization_info']['performance_metrics'] = perf_metrics
            
            # Add optimization result if present
            if 'optimization_result' in result:
                opt_result = result['optimization_result']
                metadata['optimization_info']['optimization_result'] = opt_result
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating optimization metadata: {e}")
            return {}
    
    def _estimate_data_size(self, data: Any) -> float:
        """Estimate data size in MB."""
        try:
            if hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum() / 1024 / 1024
            elif hasattr(data, 'nbytes'):
                return data.nbytes / 1024 / 1024
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def aggregate_pipeline_metadata(self, step_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metadata from multiple steps into pipeline metadata.
        
        Args:
            step_metadata_list: List of step metadata dictionaries
            
        Returns:
            Aggregated pipeline metadata
        """
        try:
            pipeline_metadata = {
                'pipeline_info': {
                    'total_steps': len(step_metadata_list),
                    'completed_steps': sum(1 for meta in step_metadata_list if meta.get('status') == 'completed'),
                    'failed_steps': sum(1 for meta in step_metadata_list if meta.get('status') == 'failed'),
                    'pipeline_timestamp': datetime.now().isoformat()
                },
                'step_summary': [],
                'execution_timeline': []
            }
            
            # Process each step metadata
            for i, step_metadata in enumerate(step_metadata_list):
                step_summary = {
                    'step_index': i,
                    'step_name': step_metadata.get('step_name', f'step_{i}'),
                    'step_type': step_metadata.get('step_type', 'unknown'),
                    'status': step_metadata.get('status', 'unknown'),
                    'execution_timestamp': step_metadata.get('execution_timestamp', 'unknown')
                }
                
                pipeline_metadata['step_summary'].append(step_summary)
                pipeline_metadata['execution_timeline'].append({
                    'step_name': step_metadata.get('step_name', f'step_{i}'),
                    'timestamp': step_metadata.get('execution_timestamp', 'unknown'),
                    'status': step_metadata.get('status', 'unknown')
                })
            
            # Add aggregated statistics
            pipeline_metadata['pipeline_info'].update({
                'success_rate': pipeline_metadata['pipeline_info']['completed_steps'] / max(1, pipeline_metadata['pipeline_info']['total_steps']),
                'step_types': list(set(meta.get('step_type', 'unknown') for meta in step_metadata_list))
            })
            
            return pipeline_metadata
            
        except Exception as e:
            self.logger.warning(f"Error aggregating pipeline metadata: {e}")
            return {
                'pipeline_info': {
                    'total_steps': len(step_metadata_list),
                    'aggregation_error': str(e),
                    'pipeline_timestamp': datetime.now().isoformat()
                }
            }
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata structure and completeness.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            validation_result = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'metadata_info': {
                    'has_required_fields': True,
                    'field_count': len(metadata),
                    'field_names': list(metadata.keys())
                }
            }
            
            # Check required fields
            required_fields = ['step_name', 'execution_timestamp', 'status']
            missing_fields = [field for field in required_fields if field not in metadata]
            
            if missing_fields:
                validation_result['errors'].append(f"Missing required fields: {missing_fields}")
                validation_result['passed'] = False
                validation_result['metadata_info']['has_required_fields'] = False
            
            # Check timestamp format
            timestamp = metadata.get('execution_timestamp', '')
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                validation_result['warnings'].append(f"Invalid timestamp format: {timestamp}")
            
            # Check status values
            valid_statuses = ['completed', 'failed', 'running', 'pending', 'cancelled']
            status = metadata.get('status', '')
            if status not in valid_statuses:
                validation_result['warnings'].append(f"Unusual status value: {status}")
            
            return validation_result
            
        except Exception as e:
            self.logger.warning(f"Error validating metadata: {e}")
            return {
                'passed': False,
                'errors': [f"Metadata validation error: {e}"],
                'warnings': [],
                'metadata_info': {}
            }
    
    def format_metadata_for_logging(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata for logging output.
        
        Args:
            metadata: Metadata dictionary to format
            
        Returns:
            Formatted metadata string
        """
        try:
            # Extract key information
            step_name = metadata.get('step_name', 'unknown')
            step_type = metadata.get('step_type', 'unknown')
            status = metadata.get('status', 'unknown')
            timestamp = metadata.get('execution_timestamp', 'unknown')
            
            # Create summary
            summary = f"Step: {step_name} | Type: {step_type} | Status: {status} | Time: {timestamp}"
            
            # Add step-specific information
            if step_type == 'data_collection' and 'data_collection_info' in metadata:
                data_info = metadata['data_collection_info']
                if 'data_shape' in data_info:
                    summary += f" | Data Shape: {data_info['data_shape']}"
            
            elif step_type == 'feature_engineering' and 'feature_engineering_info' in metadata:
                feat_info = metadata['feature_engineering_info']
                if 'feature_count' in feat_info:
                    summary += f" | Features: {feat_info['feature_count']}"
            
            elif step_type == 'model_training' and 'model_training_info' in metadata:
                model_info = metadata['model_training_info']
                if 'model_type' in model_info:
                    summary += f" | Model: {model_info['model_type']}"
            
            elif step_type == 'model_evaluation' and 'model_evaluation_info' in metadata:
                eval_info = metadata['model_evaluation_info']
                if 'primary_metric' in eval_info:
                    summary += f" | Primary Metric: {eval_info['primary_metric']:.3f}"
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Error formatting metadata for logging: {e}")
            return f"Metadata formatting error: {e}"


# Global instance for easy access
_global_metadata_utilities = None

def get_metadata_utilities(config: Optional[Dict[str, Any]] = None) -> MetadataUtilities:
    """Get metadata utilities instance."""
    global _global_metadata_utilities
    if _global_metadata_utilities is None:
        _global_metadata_utilities = MetadataUtilities(config)
    return _global_metadata_utilities


# Convenience functions
def generate_step_metadata(step_name: str, step_type: str, 
                          result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate step metadata using utilities."""
    utils = get_metadata_utilities()
    return utils.generate_step_metadata(step_name, step_type, result, config)


def aggregate_pipeline_metadata(step_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate pipeline metadata using utilities."""
    utils = get_metadata_utilities()
    return utils.aggregate_pipeline_metadata(step_metadata_list)


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate metadata using utilities."""
    utils = get_metadata_utilities()
    return utils.validate_metadata(metadata)


def format_metadata_for_logging(metadata: Dict[str, Any]) -> str:
    """Format metadata for logging using utilities."""
    utils = get_metadata_utilities()
    return utils.format_metadata_for_logging(metadata)


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
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
    
    # Test metadata utilities
    utils = MetadataUtilities()
    
    # Test step metadata generation
    print("=== Step Metadata Generation ===")
    
    # Data collection step
    data_collection_result = {'data': data, 'status': 'completed'}
    data_collection_config = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'timeframe': '1m'}
    data_metadata = utils.generate_step_metadata('data_collection', 'data_collection', data_collection_result, data_collection_config)
    print(f"Data collection metadata: {utils.format_metadata_for_logging(data_metadata)}")
    
    # Feature engineering step
    features = data.copy()
    features['returns'] = features['close'].pct_change()
    feature_engineering_result = {'features': features, 'status': 'completed'}
    feature_metadata = utils.generate_step_metadata('feature_engineering', 'feature_engineering', feature_engineering_result, data_collection_config)
    print(f"Feature engineering metadata: {utils.format_metadata_for_logging(feature_metadata)}")
    
    # Model training step
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    targets = pd.Series(np.random.randint(0, 2, len(features)), name='target')
    model.fit(features, targets)
    model_training_result = {'model': model, 'evaluation_metrics': {'accuracy': 0.85}, 'status': 'completed'}
    training_metadata = utils.generate_step_metadata('model_training', 'model_training', model_training_result, data_collection_config)
    print(f"Model training metadata: {utils.format_metadata_for_logging(training_metadata)}")
    
    # Model evaluation step
    predictions = model.predict(features)
    model_evaluation_result = {'evaluation_metrics': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88}, 'predictions': predictions, 'status': 'completed'}
    evaluation_metadata = utils.generate_step_metadata('model_evaluation', 'model_evaluation', model_evaluation_result, data_collection_config)
    print(f"Model evaluation metadata: {utils.format_metadata_for_logging(evaluation_metadata)}")
    
    # Test pipeline metadata aggregation
    print("\n=== Pipeline Metadata Aggregation ===")
    step_metadata_list = [data_metadata, feature_metadata, training_metadata, evaluation_metadata]
    pipeline_metadata = utils.aggregate_pipeline_metadata(step_metadata_list)
    print(f"Pipeline info: {pipeline_metadata['pipeline_info']}")
    print(f"Success rate: {pipeline_metadata['pipeline_info']['success_rate']:.2%}")
    
    # Test metadata validation
    print("\n=== Metadata Validation ===")
    validation_result = utils.validate_metadata(data_metadata)
    print(f"Validation passed: {validation_result['passed']}")
    print(f"Errors: {validation_result['errors']}")
    print(f"Warnings: {validation_result['warnings']}")