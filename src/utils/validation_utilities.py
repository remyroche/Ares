"""
Validation Utilities

This module provides validation utilities extracted from training steps
to eliminate code duplication and provide consistent validation across all steps.

Key Features:
- Input/output validation patterns
- Data type validation
- Range and format validation
- Validation result aggregation
- Integration with pipeline infrastructure
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class ValidationUtilities:
    """
    Validation utilities for all training steps.
    
    This provides common validation patterns and utilities
    extracted from multiple training step implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validation utilities."""
        self.config = config or {}
        self.logger = logger.getChild('ValidationUtilities')
        
        self.logger.info("🚀 Validation Utilities initialized")
    
    def validate_input_data(self, data: Any, data_type: str = 'general', 
                           required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate input data for training steps.
        
        Args:
            data: Data to validate
            data_type: Type of data ('ohlcv', 'features', 'targets', 'general')
            required_columns: List of required column names
            
        Returns:
            Validation result dictionary
        """
        try:
            self.logger.info(f"🔍 Validating input data (type: {data_type})")
            
            validation_result = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'data_type': data_type,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Check if data is None or empty
            if data is None:
                validation_result['errors'].append("Data is None")
                validation_result['passed'] = False
                return validation_result
            
            # Convert to DataFrame if needed
            df = self._ensure_dataframe(data)
            if df is None:
                validation_result['errors'].append("Could not convert data to DataFrame")
                validation_result['passed'] = False
                return validation_result
            
            # Check if DataFrame is empty
            if df.empty:
                validation_result['errors'].append("DataFrame is empty")
                validation_result['passed'] = False
                return validation_result
            
            # Add basic data info
            validation_result['data_info'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
            
            # Validate required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                    validation_result['passed'] = False
            
            # Data type specific validation
            if data_type == 'ohlcv':
                validation_result.update(self._validate_ohlcv_data(df))
            elif data_type == 'features':
                validation_result.update(self._validate_features_data(df))
            elif data_type == 'targets':
                validation_result.update(self._validate_targets_data(df))
            
            # General validation
            validation_result.update(self._validate_general_data(df))
            
            if validation_result['passed']:
                self.logger.info(f"✅ Input data validation passed for {data_type}")
            else:
                self.logger.error(f"❌ Input data validation failed for {data_type}: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Error validating input data: {e}")
            return {
                'passed': False,
                'errors': [f"Input data validation error: {e}"],
                'warnings': [],
                'data_type': data_type,
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def validate_output_data(self, data: Any, expected_type: str = 'general',
                           input_validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output data from training steps.
        
        Args:
            data: Output data to validate
            expected_type: Expected type of output data
            input_validation: Input validation result for comparison
            
        Returns:
            Validation result dictionary
        """
        try:
            self.logger.info(f"🔍 Validating output data (type: {expected_type})")
            
            validation_result = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'expected_type': expected_type,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Check if data is None
            if data is None:
                validation_result['errors'].append("Output data is None")
                validation_result['passed'] = False
                return validation_result
            
            # Convert to DataFrame if needed
            df = self._ensure_dataframe(data)
            if df is None:
                validation_result['errors'].append("Could not convert output data to DataFrame")
                validation_result['passed'] = False
                return validation_result
            
            # Add output data info
            validation_result['output_info'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
            
            # Compare with input if available
            if input_validation and 'data_info' in input_validation:
                input_info = input_validation['data_info']
                comparison_result = self._compare_input_output(input_info, validation_result['output_info'])
                validation_result.update(comparison_result)
            
            # Type-specific output validation
            if expected_type == 'features':
                validation_result.update(self._validate_features_output(df))
            elif expected_type == 'predictions':
                validation_result.update(self._validate_predictions_output(df))
            elif expected_type == 'model':
                validation_result.update(self._validate_model_output(data))
            
            if validation_result['passed']:
                self.logger.info(f"✅ Output data validation passed for {expected_type}")
            else:
                self.logger.error(f"❌ Output data validation failed for {expected_type}: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Error validating output data: {e}")
            return {
                'passed': False,
                'errors': [f"Output data validation error: {e}"],
                'warnings': [],
                'expected_type': expected_type,
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def _ensure_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        """Ensure data is a DataFrame."""
        try:
            if data is None:
                return None
            
            if isinstance(data, pd.DataFrame):
                return data
            
            if isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            
            if isinstance(data, (list, tuple)):
                return pd.DataFrame(data)
            
            # Try to convert other types
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.warning(f"Could not convert data to DataFrame: {e}")
            return None
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate OHLCV data."""
        validation_result = {'ohlcv_validation': {}}
        
        try:
            required_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_ohlcv_columns = [col for col in required_ohlcv_columns if col not in df.columns]
            
            if missing_ohlcv_columns:
                validation_result['errors'].append(f"Missing OHLCV columns: {missing_ohlcv_columns}")
                validation_result['passed'] = False
            
            # Validate OHLC relationships
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = (
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                ).sum()
                
                if invalid_ohlc > 0:
                    validation_result['warnings'].append(f"Found {invalid_ohlc} rows with invalid OHLC relationships")
                    validation_result['ohlcv_validation']['invalid_ohlc_count'] = invalid_ohlc
            
            # Validate volume
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
                    validation_result['warnings'].append(f"Found {negative_volume} rows with negative volume")
                    validation_result['ohlcv_validation']['negative_volume_count'] = negative_volume
            
            validation_result['ohlcv_validation']['validation_passed'] = len(validation_result.get('errors', [])) == 0
            
        except Exception as e:
            self.logger.warning(f"Error validating OHLCV data: {e}")
            validation_result['warnings'].append(f"OHLCV validation error: {e}")
        
        return validation_result
    
    def _validate_features_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate features data."""
        validation_result = {'features_validation': {}}
        
        try:
            # Check for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            
            validation_result['features_validation']['numeric_columns'] = len(numeric_columns)
            validation_result['features_validation']['non_numeric_columns'] = len(non_numeric_columns)
            
            if len(non_numeric_columns) > 0:
                validation_result['warnings'].append(f"Found {len(non_numeric_columns)} non-numeric columns in features")
            
            # Check for constant features
            constant_features = []
            for col in numeric_columns:
                if df[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                validation_result['warnings'].append(f"Found {len(constant_features)} constant features")
                validation_result['features_validation']['constant_features'] = constant_features
            
            # Check for high correlation
            if len(numeric_columns) > 1:
                correlation_matrix = df[numeric_columns].corr()
                high_corr_pairs = []
                
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = abs(correlation_matrix.iloc[i, j])
                        if corr_value > 0.95:
                            high_corr_pairs.append({
                                'col1': correlation_matrix.columns[i],
                                'col2': correlation_matrix.columns[j],
                                'correlation': corr_value
                            })
                
                if high_corr_pairs:
                    validation_result['warnings'].append(f"Found {len(high_corr_pairs)} high correlation pairs")
                    validation_result['features_validation']['high_correlation_pairs'] = high_corr_pairs
            
            validation_result['features_validation']['validation_passed'] = len(validation_result.get('errors', [])) == 0
            
        except Exception as e:
            self.logger.warning(f"Error validating features data: {e}")
            validation_result['warnings'].append(f"Features validation error: {e}")
        
        return validation_result
    
    def _validate_targets_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate targets data."""
        validation_result = {'targets_validation': {}}
        
        try:
            # Check if targets are numeric
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                validation_result['warnings'].append("No numeric columns found in targets")
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                validation_result['warnings'].append(f"Found {missing_values} missing values in targets")
                validation_result['targets_validation']['missing_values'] = missing_values
            
            # Check target distribution for classification
            for col in numeric_columns:
                unique_values = df[col].nunique()
                if unique_values <= 10:  # Likely classification
                    value_counts = df[col].value_counts()
                    min_count = value_counts.min()
                    max_count = value_counts.max()
                    
                    if min_count / max_count < 0.1:  # Imbalanced classes
                        validation_result['warnings'].append(f"Imbalanced classes in {col}: min={min_count}, max={max_count}")
                        validation_result['targets_validation']['imbalanced_classes'] = {
                            'column': col,
                            'min_count': min_count,
                            'max_count': max_count,
                            'balance_ratio': min_count / max_count
                        }
            
            validation_result['targets_validation']['validation_passed'] = len(validation_result.get('errors', [])) == 0
            
        except Exception as e:
            self.logger.warning(f"Error validating targets data: {e}")
            validation_result['warnings'].append(f"Targets validation error: {e}")
        
        return validation_result
    
    def _validate_general_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate general data properties."""
        validation_result = {'general_validation': {}}
        
        try:
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            validation_result['general_validation']['missing_ratio'] = missing_ratio
            
            if missing_ratio > 0.1:  # 10% threshold
                validation_result['warnings'].append(f"High missing data ratio: {missing_ratio:.3f}")
            
            # Check for duplicates
            duplicate_ratio = df.duplicated().sum() / len(df)
            validation_result['general_validation']['duplicate_ratio'] = duplicate_ratio
            
            if duplicate_ratio > 0.05:  # 5% threshold
                validation_result['warnings'].append(f"High duplicate ratio: {duplicate_ratio:.3f}")
            
            # Check data types
            object_columns = df.select_dtypes(include=['object']).columns
            if len(object_columns) > 0:
                validation_result['warnings'].append(f"Found {len(object_columns)} object columns")
                validation_result['general_validation']['object_columns'] = list(object_columns)
            
            validation_result['general_validation']['validation_passed'] = len(validation_result.get('errors', [])) == 0
            
        except Exception as e:
            self.logger.warning(f"Error validating general data: {e}")
            validation_result['warnings'].append(f"General validation error: {e}")
        
        return validation_result
    
    def _validate_features_output(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate features output."""
        validation_result = {'features_output_validation': {}}
        
        try:
            # Check if output has more features than input
            if 'input_info' in validation_result:
                input_shape = validation_result['input_info']['shape']
                if df.shape[1] <= input_shape[1]:
                    validation_result['warnings'].append("Output features count is not greater than input")
            
            # Check for NaN values in output
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                validation_result['warnings'].append(f"Found {nan_count} NaN values in output features")
                validation_result['features_output_validation']['nan_count'] = nan_count
            
            validation_result['features_output_validation']['validation_passed'] = len(validation_result.get('errors', [])) == 0
            
        except Exception as e:
            self.logger.warning(f"Error validating features output: {e}")
            validation_result['warnings'].append(f"Features output validation error: {e}")
        
        return validation_result
    
    def _validate_predictions_output(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate predictions output."""
        validation_result = {'predictions_output_validation': {}}
        
        try:
            # Check if predictions are numeric
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                validation_result['errors'].append("Predictions must be numeric")
                validation_result['passed'] = False
            
            # Check for NaN values
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                validation_result['errors'].append(f"Found {nan_count} NaN values in predictions")
                validation_result['passed'] = False
            
            validation_result['predictions_output_validation']['validation_passed'] = len(validation_result.get('errors', [])) == 0
            
        except Exception as e:
            self.logger.warning(f"Error validating predictions output: {e}")
            validation_result['warnings'].append(f"Predictions output validation error: {e}")
        
        return validation_result
    
    def _validate_model_output(self, model: Any) -> Dict[str, Any]:
        """Validate model output."""
        validation_result = {'model_output_validation': {}}
        
        try:
            # Check if model has required methods
            required_methods = ['predict', 'fit']
            missing_methods = [method for method in required_methods if not hasattr(model, method)]
            
            if missing_methods:
                validation_result['errors'].append(f"Model missing required methods: {missing_methods}")
                validation_result['passed'] = False
            
            # Get model information
            model_info = {
                'type': type(model).__name__,
                'has_predict_proba': hasattr(model, 'predict_proba'),
                'has_feature_importances_': hasattr(model, 'feature_importances_')
            }
            
            if hasattr(model, 'get_params'):
                model_info['params'] = model.get_params()
            
            validation_result['model_output_validation']['model_info'] = model_info
            validation_result['model_output_validation']['validation_passed'] = len(validation_result.get('errors', [])) == 0
            
        except Exception as e:
            self.logger.warning(f"Error validating model output: {e}")
            validation_result['warnings'].append(f"Model output validation error: {e}")
        
        return validation_result
    
    def _compare_input_output(self, input_info: Dict[str, Any], output_info: Dict[str, Any]) -> Dict[str, Any]:
        """Compare input and output data."""
        comparison_result = {'input_output_comparison': {}}
        
        try:
            # Compare shapes
            input_shape = input_info.get('shape', (0, 0))
            output_shape = output_info.get('shape', (0, 0))
            
            comparison_result['input_output_comparison']['shape_comparison'] = {
                'input_shape': input_shape,
                'output_shape': output_shape,
                'rows_changed': output_shape[0] - input_shape[0],
                'columns_changed': output_shape[1] - input_shape[1]
            }
            
            # Check if row count is preserved
            if input_shape[0] != output_shape[0]:
                comparison_result['warnings'].append(f"Row count changed from {input_shape[0]} to {output_shape[0]}")
            
            # Compare column types
            input_dtypes = input_info.get('dtypes', {})
            output_dtypes = output_info.get('dtypes', {})
            
            comparison_result['input_output_comparison']['dtype_comparison'] = {
                'input_dtypes': input_dtypes,
                'output_dtypes': output_dtypes
            }
            
        except Exception as e:
            self.logger.warning(f"Error comparing input and output: {e}")
            comparison_result['warnings'].append(f"Input/output comparison error: {e}")
        
        return comparison_result
    
    def aggregate_validation_results(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple validation results.
        
        Args:
            validation_results: List of validation result dictionaries
            
        Returns:
            Aggregated validation result
        """
        try:
            aggregated_result = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'validation_count': len(validation_results),
                'passed_count': 0,
                'failed_count': 0,
                'aggregation_timestamp': datetime.now().isoformat()
            }
            
            # Aggregate results
            for i, result in enumerate(validation_results):
                if result.get('passed', False):
                    aggregated_result['passed_count'] += 1
                else:
                    aggregated_result['failed_count'] += 1
                    aggregated_result['passed'] = False
                
                # Collect errors and warnings
                if 'errors' in result:
                    for error in result['errors']:
                        aggregated_result['errors'].append(f"Validation {i+1}: {error}")
                
                if 'warnings' in result:
                    for warning in result['warnings']:
                        aggregated_result['warnings'].append(f"Validation {i+1}: {warning}")
            
            # Calculate success rate
            aggregated_result['success_rate'] = aggregated_result['passed_count'] / max(1, aggregated_result['validation_count'])
            
            return aggregated_result
            
        except Exception as e:
            self.logger.warning(f"Error aggregating validation results: {e}")
            return {
                'passed': False,
                'errors': [f"Validation aggregation error: {e}"],
                'warnings': [],
                'validation_count': len(validation_results),
                'aggregation_timestamp': datetime.now().isoformat()
            }


# Global instance for easy access
_global_validation_utilities = None

def get_validation_utilities(config: Optional[Dict[str, Any]] = None) -> ValidationUtilities:
    """Get validation utilities instance."""
    global _global_validation_utilities
    if _global_validation_utilities is None:
        _global_validation_utilities = ValidationUtilities(config)
    return _global_validation_utilities


# Convenience functions
def validate_input_data(data: Any, data_type: str = 'general', 
                       required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate input data using utilities."""
    utils = get_validation_utilities()
    return utils.validate_input_data(data, data_type, required_columns)


def validate_output_data(data: Any, expected_type: str = 'general',
                        input_validation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate output data using utilities."""
    utils = get_validation_utilities()
    return utils.validate_output_data(data, expected_type, input_validation)


def aggregate_validation_results(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate validation results using utilities."""
    utils = get_validation_utilities()
    return utils.aggregate_validation_results(validation_results)


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    ohlcv_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Add some issues
    ohlcv_data.iloc[10:20, 1] = np.nan  # Missing values
    ohlcv_data.iloc[100:105, :] = ohlcv_data.iloc[50:55, :].values  # Duplicates
    
    # Test validation utilities
    utils = ValidationUtilities()
    
    print("=== Input Data Validation ===")
    input_validation = utils.validate_input_data(ohlcv_data, 'ohlcv')
    print(f"Input validation passed: {input_validation['passed']}")
    print(f"Errors: {input_validation['errors']}")
    print(f"Warnings: {input_validation['warnings']}")
    
    print("\n=== Features Data Validation ===")
    features_data = ohlcv_data.copy()
    features_data['returns'] = features_data['close'].pct_change()
    features_data['volatility'] = features_data['returns'].rolling(20).std()
    
    features_validation = utils.validate_input_data(features_data, 'features')
    print(f"Features validation passed: {features_validation['passed']}")
    print(f"Warnings: {features_validation['warnings']}")
    
    print("\n=== Targets Data Validation ===")
    targets_data = pd.DataFrame({
        'target': np.random.randint(0, 2, len(features_data))
    })
    
    targets_validation = utils.validate_input_data(targets_data, 'targets')
    print(f"Targets validation passed: {targets_validation['passed']}")
    print(f"Warnings: {targets_validation['warnings']}")
    
    print("\n=== Output Data Validation ===")
    output_validation = utils.validate_output_data(features_data, 'features', input_validation)
    print(f"Output validation passed: {output_validation['passed']}")
    print(f"Warnings: {output_validation['warnings']}")
    
    print("\n=== Validation Results Aggregation ===")
    validation_results = [input_validation, features_validation, targets_validation, output_validation]
    aggregated_result = utils.aggregate_validation_results(validation_results)
    print(f"Aggregated validation passed: {aggregated_result['passed']}")
    print(f"Success rate: {aggregated_result['success_rate']:.2%}")
    print(f"Total errors: {len(aggregated_result['errors'])}")
    print(f"Total warnings: {len(aggregated_result['warnings'])}")