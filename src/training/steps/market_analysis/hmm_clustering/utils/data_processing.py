"""
Data processing utilities for HMM clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processing utilities with hardware acceleration."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the data processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for data processing")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for data processing: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for data processing")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for data processing: {e}")

    def clean_data(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[Union[pd.DataFrame, np.ndarray], Dict[str, Any]]:
        """Clean and preprocess data.

        Args:
            data: Input data

        Returns:
            Tuple of (cleaned_data, cleaning_metadata)
        """
        try:
            if isinstance(data, pd.DataFrame):
                return self._clean_dataframe(data)
            else:
                return self._clean_array(data)
                
        except Exception as e:
            self.logger.error(f"❌ Data cleaning failed: {e}")
            return data, {'error': str(e)}

    def _clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (cleaned_df, cleaning_metadata)
        """
        try:
            original_shape = df.shape
            cleaning_metadata = {
                'original_shape': original_shape,
                'cleaning_steps': []
            }
            
            # Remove duplicate rows
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed_duplicates = initial_rows - len(df)
            if removed_duplicates > 0:
                cleaning_metadata['cleaning_steps'].append(f'Removed {removed_duplicates} duplicate rows')
            
            # Handle missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                # Fill missing values
                df = df.fillna(df.median())
                cleaning_metadata['cleaning_steps'].append(f'Filled {missing_counts.sum()} missing values')
            
            # Remove infinite values
            inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
            if inf_mask.any().any():
                df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median())
                cleaning_metadata['cleaning_steps'].append('Replaced infinite values')
            
            # Convert to numeric where possible
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
            
            cleaning_metadata['final_shape'] = df.shape
            cleaning_metadata['success'] = True
            
            return df, cleaning_metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ DataFrame cleaning failed: {e}")
            return df, {'error': str(e)}

    def _clean_array(self, arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Clean numpy array.

        Args:
            arr: Input array

        Returns:
            Tuple of (cleaned_array, cleaning_metadata)
        """
        try:
            original_shape = arr.shape
            cleaning_metadata = {
                'original_shape': original_shape,
                'cleaning_steps': []
            }
            
            # Handle NaN values
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                arr = np.nan_to_num(arr, nan=0.0)
                cleaning_metadata['cleaning_steps'].append(f'Replaced {nan_count} NaN values')
            
            # Handle infinite values
            inf_count = np.isinf(arr).sum()
            if inf_count > 0:
                arr = np.nan_to_num(arr, posinf=0.0, neginf=0.0)
                cleaning_metadata['cleaning_steps'].append(f'Replaced {inf_count} infinite values')
            
            cleaning_metadata['final_shape'] = arr.shape
            cleaning_metadata['success'] = True
            
            return arr, cleaning_metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ Array cleaning failed: {e}")
            return arr, {'error': str(e)}

    def validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Validate data quality.

        Args:
            data: Input data

        Returns:
            Validation results
        """
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'statistics': {}
            }
            
            if isinstance(data, pd.DataFrame):
                validation_results.update(self._validate_dataframe(data))
            else:
                validation_results.update(self._validate_array(data))
            
            return validation_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Data validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'error': str(e)
            }

    def _validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Validation results
        """
        try:
            validation_results = {
                'data_type': 'DataFrame',
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'issues': [],
                'statistics': {}
            }
            
            # Check for empty DataFrame
            if df.empty:
                validation_results['issues'].append('DataFrame is empty')
                validation_results['is_valid'] = False
            
            # Check for missing values
            missing_counts = df.isnull().sum()
            if missing_counts.sum() > 0:
                validation_results['issues'].append(f'Missing values: {missing_counts.sum()}')
            
            # Check for infinite values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                inf_counts = np.isinf(df[numeric_cols]).sum()
                if inf_counts.sum() > 0:
                    validation_results['issues'].append(f'Infinite values: {inf_counts.sum()}')
            
            # Calculate basic statistics
            if len(numeric_cols) > 0:
                validation_results['statistics'] = df[numeric_cols].describe().to_dict()
            
            # Check data quality
            if len(validation_results['issues']) > 0:
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ DataFrame validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'error': str(e)
            }

    def _validate_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Validate numpy array.

        Args:
            arr: Input array

        Returns:
            Validation results
        """
        try:
            validation_results = {
                'data_type': 'numpy_array',
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'issues': [],
                'statistics': {}
            }
            
            # Check for empty array
            if arr.size == 0:
                validation_results['issues'].append('Array is empty')
                validation_results['is_valid'] = False
                return validation_results
            
            # Check for NaN values
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                validation_results['issues'].append(f'NaN values: {nan_count}')
            
            # Check for infinite values
            inf_count = np.isinf(arr).sum()
            if inf_count > 0:
                validation_results['issues'].append(f'Infinite values: {inf_count}')
            
            # Calculate basic statistics
            if arr.size > 0:
                validation_results['statistics'] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr))
                }
            
            # Check data quality
            if len(validation_results['issues']) > 0:
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Array validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'error': str(e)
            }


def create_data_processor(config: Dict[str, Any] = None) -> DataProcessor:
    """Create a data processor instance.

    Args:
        config: Configuration dictionary

    Returns:
        DataProcessor instance
    """
    return DataProcessor(config)