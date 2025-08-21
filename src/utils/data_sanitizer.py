"""
Data Sanitizer for Enhanced Training Manager
Provides data sanitization and cleaning functionality
"""

import logging
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataSanitizer:
    """Sanitizes and cleans data for training pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataSanitizer")
        self.sanitization_results = {}
    
    def sanitize_identifier(self, identifier: str) -> str:
        """Sanitize an identifier (symbol, exchange, etc.) for safe file operations"""
        if not isinstance(identifier, str):
            return str(identifier)
        
        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', identifier)
        
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip('. ')
        
        # Replace multiple underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unknown"
        
        return sanitized
    
    def sanitize_dataframe(self, df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
        """Sanitize a pandas DataFrame"""
        try:
            sanitized_df = df.copy()
            
            # Handle infinite values
            numeric_cols = sanitized_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Replace inf with NaN
                sanitized_df[col] = sanitized_df[col].replace([np.inf, -np.inf], np.nan)
            
            # Handle extreme outliers (optional)
            for col in numeric_cols:
                if len(sanitized_df[col].dropna()) > 0:
                    q1 = sanitized_df[col].quantile(0.01)
                    q99 = sanitized_df[col].quantile(0.99)
                    sanitized_df[col] = sanitized_df[col].clip(lower=q1, upper=q99)
            
            # Ensure column names are valid
            sanitized_df.columns = [self.sanitize_column_name(col) for col in sanitized_df.columns]
            
            self.sanitization_results[name] = {
                'original_shape': df.shape,
                'sanitized_shape': sanitized_df.shape,
                'infinite_values_replaced': True,
                'outliers_clipped': True,
                'columns_sanitized': True
            }
            
            self.logger.info(f"DataFrame '{name}' sanitized successfully")
            return sanitized_df
            
        except Exception as e:
            error_msg = f"Error sanitizing DataFrame '{name}': {str(e)}"
            self.logger.error(error_msg)
            return df  # Return original if sanitization fails
    
    def sanitize_column_name(self, column_name: str) -> str:
        """Sanitize a column name for safe DataFrame operations"""
        if not isinstance(column_name, str):
            return str(column_name)
        
        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', column_name)
        
        # Remove leading/trailing whitespace
        sanitized = sanitized.strip()
        
        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')
        
        # Replace multiple underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "column"
        
        return sanitized
    
    def sanitize_training_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize training data parameters"""
        try:
            sanitized_data = data.copy()
            
            # Sanitize string parameters
            string_params = ['symbol', 'exchange', 'timeframe']
            for param in string_params:
                if param in sanitized_data and isinstance(sanitized_data[param], str):
                    sanitized_data[param] = sanitized_data[param].strip()
            
            # Sanitize numeric parameters
            numeric_params = ['lookback_days', 'max_trials', 'n_trials']
            for param in numeric_params:
                if param in sanitized_data:
                    try:
                        sanitized_data[param] = float(sanitized_data[param])
                        if sanitized_data[param] <= 0:
                            sanitized_data[param] = 1.0  # Default value
                    except (ValueError, TypeError):
                        sanitized_data[param] = 1.0  # Default value
            
            # Sanitize boolean parameters
            boolean_params = ['enable_model_training', 'blank_training_mode', 'enable_validators']
            for param in boolean_params:
                if param in sanitized_data:
                    if isinstance(sanitized_data[param], str):
                        sanitized_data[param] = sanitized_data[param].lower() in ['true', '1', 'yes']
                    elif not isinstance(sanitized_data[param], bool):
                        sanitized_data[param] = False
            
            self.sanitization_results['training_data'] = {
                'original_keys': list(data.keys()),
                'sanitized_keys': list(sanitized_data.keys()),
                'string_params_sanitized': True,
                'numeric_params_sanitized': True,
                'boolean_params_sanitized': True
            }
            
            self.logger.info("Training data sanitized successfully")
            return sanitized_data
            
        except Exception as e:
            error_msg = f"Error sanitizing training data: {str(e)}"
            self.logger.error(error_msg)
            return data  # Return original if sanitization fails
    
    def sanitize_file_path(self, file_path: Union[str, Path]) -> str:
        """Sanitize a file path for safe file operations"""
        if isinstance(file_path, Path):
            file_path = str(file_path)
        
        if not isinstance(file_path, str):
            return str(file_path)
        
        # Normalize path separators
        sanitized = file_path.replace('\\', '/')
        
        # Remove or replace unsafe characters
        sanitized = re.sub(r'[<>:"|?*]', '_', sanitized)
        
        # Remove leading/trailing whitespace
        sanitized = sanitized.strip()
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unknown_path"
        
        return sanitized
    
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration dictionary"""
        try:
            sanitized_config = config.copy()
            
            # Sanitize string values
            for key, value in sanitized_config.items():
                if isinstance(value, str):
                    sanitized_config[key] = value.strip()
                elif isinstance(value, dict):
                    sanitized_config[key] = self.sanitize_config(value)
                elif isinstance(value, list):
                    sanitized_config[key] = [
                        item.strip() if isinstance(item, str) else item 
                        for item in value
                    ]
            
            self.sanitization_results['config'] = {
                'original_keys': list(config.keys()),
                'sanitized_keys': list(sanitized_config.keys()),
                'string_values_sanitized': True
            }
            
            self.logger.info("Configuration sanitized successfully")
            return sanitized_config
            
        except Exception as e:
            error_msg = f"Error sanitizing configuration: {str(e)}"
            self.logger.error(error_msg)
            return config  # Return original if sanitization fails
    
    def get_sanitization_summary(self) -> Dict[str, Any]:
        """Get summary of all sanitization results"""
        summary = {
            'total_sanitizations': len(self.sanitization_results),
            'successful': 0,
            'failed': 0,
            'results': {}
        }
        
        for name, result in self.sanitization_results.items():
            summary['results'][name] = result
            
            # Count successful sanitizations (those with result data)
            if isinstance(result, dict) and len(result) > 0:
                summary['successful'] += 1
            else:
                summary['failed'] += 1
        
        return summary
    
    def clear_results(self):
        """Clear all sanitization results"""
        self.sanitization_results.clear()
        self.logger.info("Sanitization results cleared")