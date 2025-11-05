"""
Result Converter for Statsmodels Clustering

This module provides utilities to convert between different result formats
for statsmodels regime switching models.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


class ConversionConfig:
    """Configuration for result conversion."""
    
    def __init__(self, 
                 include_metadata: bool = True,
                 include_diagnostics: bool = True,
                 format_type: str = "unified"):
        self.include_metadata = include_metadata
        self.include_diagnostics = include_diagnostics
        self.format_type = format_type


class ConversionResult:
    """Result container for conversion operations."""
    
    def __init__(self, 
                 success: bool = True,
                 data: Optional[Dict[str, Any]] = None,
                 error_message: Optional[str] = None):
        self.success = success
        self.data = data or {}
        self.error_message = error_message


class ResultConverter:
    """Convert between different result formats for regime switching models."""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
    
    def convert_statsmodels_to_pyro(self, statsmodels_result: Any) -> ConversionResult:
        """Convert statsmodels result to Pyro format."""
        try:
            # Create a basic conversion
            converted_data = {
                "parameters": getattr(statsmodels_result, 'params', {}),
                "covariance": getattr(statsmodels_result, 'cov_params', None),
                "log_likelihood": getattr(statsmodels_result, 'llf', None),
                "aic": getattr(statsmodels_result, 'aic', None),
                "bic": getattr(statsmodels_result, 'bic', None)
            }
            
            return ConversionResult(success=True, data=converted_data)
        except Exception as e:
            return ConversionResult(success=False, error_message=str(e))
    
    def convert_pyro_to_statsmodels(self, pyro_result: Any) -> ConversionResult:
        """Convert Pyro result to statsmodels format."""
        try:
            # Create a basic conversion
            converted_data = {
                "params": getattr(pyro_result, 'params', {}),
                "cov_params": getattr(pyro_result, 'covariance', None),
                "llf": getattr(pyro_result, 'log_likelihood', None),
                "aic": getattr(pyro_result, 'aic', None),
                "bic": getattr(pyro_result, 'bic', None)
            }
            
            return ConversionResult(success=True, data=converted_data)
        except Exception as e:
            return ConversionResult(success=False, error_message=str(e))
    
    def create_unified_result(self, result: Any, source_format: str = "statsmodels") -> ConversionResult:
        """Create a unified result format from any input format."""
        try:
            unified_data = {
                "source_format": source_format,
                "timestamp": pd.Timestamp.now(),
                "data": result
            }
            
            return ConversionResult(success=True, data=unified_data)
        except Exception as e:
            return ConversionResult(success=False, error_message=str(e))
    
    def save_result_to_file(self, result: ConversionResult, filepath: str) -> bool:
        """Save result to file."""
        try:
            import json
            
            with open(filepath, 'w') as f:
                json.dump(result.data, f, indent=2, default=str)
            
            return True
        except Exception:
            return False


def convert_statsmodels_to_pyro(statsmodels_result: Any, config: Optional[ConversionConfig] = None) -> ConversionResult:
    """Convenience function to convert statsmodels result to Pyro format."""
    converter = ResultConverter(config)
    return converter.convert_statsmodels_to_pyro(statsmodels_result)


def convert_pyro_to_statsmodels(pyro_result: Any, config: Optional[ConversionConfig] = None) -> ConversionResult:
    """Convenience function to convert Pyro result to statsmodels format."""
    converter = ResultConverter(config)
    return converter.convert_pyro_to_statsmodels(pyro_result)


def create_unified_result(result: Any, source_format: str = "statsmodels", config: Optional[ConversionConfig] = None) -> ConversionResult:
    """Convenience function to create a unified result format."""
    converter = ResultConverter(config)
    return converter.create_unified_result(result, source_format)


def save_result_to_file(result: ConversionResult, filepath: str) -> bool:
    """Convenience function to save result to file."""
    converter = ResultConverter()
    return converter.save_result_to_file(result, filepath)