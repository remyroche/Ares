#!/usr/bin/env python3
"""
Enhanced Step Validator

This module provides comprehensive validation for each step in the market analysis pipeline,
ensuring data quality, schema validation, and proper step transitions.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core decorators and utilities
from src.core.decorators import (
    handles_errors,
    traced,
    validates,
    log_execution_time,
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    safe_json_load,
    validate_dataframe,
    validate_data_quality,
    get_logger,
)
from src.utils.data_quality_framework import DataQualityFramework


class EnhancedStepValidator:
    """
    Enhanced validator for market analysis pipeline steps with comprehensive
    validation capabilities including data quality, schema validation, and
    step transition validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced step validator."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.data_quality_framework = DataQualityFramework()
        
        # Validation schemas for each step
        self.step_schemas = {
            'hmm_clustering': {
                'input_files': [
                    'aggtrades_{exchange}_{symbol}_consolidated.parquet',
                    'volume_{exchange}_{symbol}_consolidated.parquet',
                ],
                'output_files': [
                    'hmm_regimes_{symbol}_{timeframe}.parquet',
                    'hmm_model_{symbol}_{timeframe}.pkl',
                ],
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'data_quality_thresholds': {
                    'max_nan_ratio': 0.05,
                    'min_data_points': 1000,
                },
            },
            'regime_splitting': {
                'input_files': [
                    'hmm_regimes_{symbol}_{timeframe}.parquet',
                    'aggtrades_{exchange}_{symbol}_consolidated.parquet',
                ],
                'output_files': [
                    'regime_data_{symbol}_{timeframe}.parquet',
                    'regime_labels_{symbol}_{timeframe}.json',
                ],
                'required_columns': ['regime', 'open', 'high', 'low', 'close', 'volume'],
                'data_quality_thresholds': {
                    'max_nan_ratio': 0.05,
                    'min_regimes': 2,
                },
            },
            'labeling': {
                'input_files': [
                    'regime_data_{symbol}_{timeframe}.parquet',
                ],
                'output_files': [
                    'labeled_data_{symbol}_{timeframe}.parquet',
                    'label_statistics_{symbol}_{timeframe}.json',
                ],
                'required_columns': ['regime', 'open', 'high', 'low', 'close', 'volume'],
                'data_quality_thresholds': {
                    'max_nan_ratio': 0.05,
                    'min_labels_per_regime': 100,
                },
            },
            'feature_engineering': {
                'input_files': [
                    'labeled_data_{symbol}_{timeframe}.parquet',
                ],
                'output_files': [
                    'features_{symbol}_{timeframe}.parquet',
                    'feature_metadata_{symbol}_{timeframe}.json',
                ],
                'required_columns': ['regime', 'label', 'open', 'high', 'low', 'close', 'volume'],
                'data_quality_thresholds': {
                    'max_nan_ratio': 0.1,
                    'min_features': 50,
                },
            },
            'matrix_operations': {
                'input_files': [
                    'features_{symbol}_{timeframe}.parquet',
                ],
                'output_files': [
                    'matrix_features_{symbol}_{timeframe}.parquet',
                    'matrix_metadata_{symbol}_{timeframe}.json',
                ],
                'required_columns': ['regime', 'label'],
                'data_quality_thresholds': {
                    'max_nan_ratio': 0.1,
                    'min_matrix_features': 20,
                },
            },
            'feature_selection': {
                'input_files': [
                    'matrix_features_{symbol}_{timeframe}.parquet',
                ],
                'output_files': [
                    'selected_features_{symbol}_{timeframe}.parquet',
                    'feature_importance_{symbol}_{timeframe}.json',
                ],
                'required_columns': ['regime', 'label'],
                'data_quality_thresholds': {
                    'max_nan_ratio': 0.1,
                    'min_selected_features': 10,
                },
            },
        }

    @handles_errors(Exception, fallback=False)
    @traced(operation_name="validate_step_input")
    @log_execution_time
    async def validate_step_input(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
    ) -> Dict[str, Any]:
        """
        Validate input data for a specific step.
        
        Args:
            step_name: Name of the step to validate
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory path
            
        Returns:
            Dict containing validation results
        """
        self.logger.info(f"🔍 Validating input for step: {step_name}")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'input_files': [],
            'data_quality': {},
        }
        
        try:
            # Get step schema
            schema = self.step_schemas.get(step_name)
            if not schema:
                validation_result['errors'].append(f"No schema found for step: {step_name}")
                validation_result['valid'] = False
                return validation_result
            
            # Validate input files exist
            data_path = Path(data_dir)
            for file_pattern in schema['input_files']:
                file_name = file_pattern.format(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                file_path = data_path / file_name
                
                if safe_file_exists(file_path):
                    validation_result['input_files'].append(str(file_path))
                else:
                    validation_result['errors'].append(f"Required input file not found: {file_path}")
                    validation_result['valid'] = False
            
            # Validate data quality for each input file
            for file_path in validation_result['input_files']:
                if file_path.endswith('.parquet'):
                    file_quality = await self._validate_file_quality(
                        file_path, schema, step_name
                    )
                    validation_result['data_quality'][file_path] = file_quality
                    
                    if not file_quality['valid']:
                        validation_result['errors'].extend(file_quality['errors'])
                        validation_result['valid'] = False
                    else:
                        validation_result['warnings'].extend(file_quality['warnings'])
            
            if validation_result['valid']:
                self.logger.info(f"✅ Input validation passed for step: {step_name}")
            else:
                self.logger.error(f"❌ Input validation failed for step: {step_name}")
                self.logger.error(f"Errors: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"❌ Input validation failed for step {step_name}: {e}")
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation exception: {str(e)}")
            return validation_result

    @handles_errors(Exception, fallback=False)
    @traced(operation_name="validate_step_output")
    @log_execution_time
    async def validate_step_output(
        self,
        step_name: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
    ) -> Dict[str, Any]:
        """
        Validate output data for a specific step.
        
        Args:
            step_name: Name of the step to validate
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory path
            
        Returns:
            Dict containing validation results
        """
        self.logger.info(f"🔍 Validating output for step: {step_name}")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'output_files': [],
            'data_quality': {},
        }
        
        try:
            # Get step schema
            schema = self.step_schemas.get(step_name)
            if not schema:
                validation_result['errors'].append(f"No schema found for step: {step_name}")
                validation_result['valid'] = False
                return validation_result
            
            # Validate output files exist
            data_path = Path(data_dir)
            for file_pattern in schema['output_files']:
                file_name = file_pattern.format(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                file_path = data_path / file_name
                
                if safe_file_exists(file_path):
                    validation_result['output_files'].append(str(file_path))
                else:
                    validation_result['errors'].append(f"Required output file not found: {file_path}")
                    validation_result['valid'] = False
            
            # Validate data quality for each output file
            for file_path in validation_result['output_files']:
                if file_path.endswith('.parquet'):
                    file_quality = await self._validate_file_quality(
                        file_path, schema, step_name
                    )
                    validation_result['data_quality'][file_path] = file_quality
                    
                    if not file_quality['valid']:
                        validation_result['errors'].extend(file_quality['errors'])
                        validation_result['valid'] = False
                    else:
                        validation_result['warnings'].extend(file_quality['warnings'])
            
            if validation_result['valid']:
                self.logger.info(f"✅ Output validation passed for step: {step_name}")
            else:
                self.logger.error(f"❌ Output validation failed for step: {step_name}")
                self.logger.error(f"Errors: {validation_result['errors']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"❌ Output validation failed for step {step_name}: {e}")
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation exception: {str(e)}")
            return validation_result

    @handles_errors(Exception, fallback=False)
    async def _validate_file_quality(
        self,
        file_path: str,
        schema: Dict[str, Any],
        step_name: str,
    ) -> Dict[str, Any]:
        """Validate the quality of a specific file."""
        quality_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {},
        }
        
        try:
            import pandas as pd
            
            # Read the file
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                # Skip non-parquet files for now
                return quality_result
            
            # Basic file info
            quality_result['file_info'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'file_size_mb': Path(file_path).stat().st_size / 1024 / 1024,
            }
            
            # Check if file is empty
            if df.empty:
                quality_result['errors'].append("File is empty")
                quality_result['valid'] = False
                return quality_result
            
            # Validate required columns
            required_columns = schema.get('required_columns', [])
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                quality_result['errors'].append(f"Missing required columns: {missing_columns}")
                quality_result['valid'] = False
            
            # Data quality validation
            quality_thresholds = schema.get('data_quality_thresholds', {})
            
            # Check NaN ratio
            max_nan_ratio = quality_thresholds.get('max_nan_ratio', 0.1)
            nan_ratios = df.isna().sum() / len(df)
            high_nan_cols = nan_ratios[nan_ratios > max_nan_ratio]
            if not high_nan_cols.empty:
                quality_result['warnings'].append(
                    f"High NaN ratio in columns: {high_nan_cols.to_dict()}"
                )
            
            # Check minimum data points
            min_data_points = quality_thresholds.get('min_data_points', 100)
            if len(df) < min_data_points:
                quality_result['errors'].append(
                    f"Insufficient data points: {len(df)} < {min_data_points}"
                )
                quality_result['valid'] = False
            
            # Step-specific validations
            await self._validate_step_specific_quality(
                step_name, df, quality_result, quality_thresholds
            )
            
            return quality_result
            
        except Exception as e:
            self.logger.exception(f"❌ File quality validation failed for {file_path}: {e}")
            quality_result['valid'] = False
            quality_result['errors'].append(f"Quality validation exception: {str(e)}")
            return quality_result

    @handles_errors(Exception, fallback=None)
    async def _validate_step_specific_quality(
        self,
        step_name: str,
        df: 'pd.DataFrame',
        quality_result: Dict[str, Any],
        quality_thresholds: Dict[str, Any],
    ) -> None:
        """Perform step-specific quality validations."""
        try:
            if step_name == 'hmm_clustering':
                # Validate HMM clustering results
                if 'regime' in df.columns:
                    unique_regimes = df['regime'].nunique()
                    min_regimes = quality_thresholds.get('min_regimes', 2)
                    if unique_regimes < min_regimes:
                        quality_result['warnings'].append(
                            f"Low number of regimes: {unique_regimes} < {min_regimes}"
                        )
            
            elif step_name == 'labeling':
                # Validate labeling results
                if 'label' in df.columns:
                    label_counts = df['label'].value_counts()
                    min_labels_per_regime = quality_thresholds.get('min_labels_per_regime', 100)
                    
                    for regime, count in label_counts.items():
                        if count < min_labels_per_regime:
                            quality_result['warnings'].append(
                                f"Low label count for regime {regime}: {count} < {min_labels_per_regime}"
                            )
            
            elif step_name == 'feature_engineering':
                # Validate feature engineering results
                feature_columns = [col for col in df.columns if col not in ['regime', 'label', 'open', 'high', 'low', 'close', 'volume']]
                min_features = quality_thresholds.get('min_features', 50)
                
                if len(feature_columns) < min_features:
                    quality_result['warnings'].append(
                        f"Low number of features: {len(feature_columns)} < {min_features}"
                    )
            
            elif step_name == 'matrix_operations':
                # Validate matrix operations results
                matrix_columns = [col for col in df.columns if col not in ['regime', 'label']]
                min_matrix_features = quality_thresholds.get('min_matrix_features', 20)
                
                if len(matrix_columns) < min_matrix_features:
                    quality_result['warnings'].append(
                        f"Low number of matrix features: {len(matrix_columns)} < {min_matrix_features}"
                    )
            
            elif step_name == 'feature_selection':
                # Validate feature selection results
                selected_columns = [col for col in df.columns if col not in ['regime', 'label']]
                min_selected_features = quality_thresholds.get('min_selected_features', 10)
                
                if len(selected_columns) < min_selected_features:
                    quality_result['warnings'].append(
                        f"Low number of selected features: {len(selected_columns)} < {min_selected_features}"
                    )
        
        except Exception as e:
            self.logger.warning(f"⚠️ Step-specific validation failed for {step_name}: {e}")

    @handles_errors(Exception, fallback=False)
    @traced(operation_name="validate_step_transition")
    @log_execution_time
    async def validate_step_transition(
        self,
        from_step: str,
        to_step: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
    ) -> Dict[str, Any]:
        """
        Validate the transition from one step to another.
        
        Args:
            from_step: Source step name
            to_step: Target step name
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory path
            
        Returns:
            Dict containing validation results
        """
        self.logger.info(f"🔍 Validating transition from {from_step} to {to_step}")
        
        transition_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'transition_info': {},
        }
        
        try:
            # Validate that the from_step output exists and is valid
            from_output_validation = await self.validate_step_output(
                from_step, symbol, exchange, timeframe, data_dir
            )
            
            if not from_output_validation['valid']:
                transition_result['errors'].extend(from_output_validation['errors'])
                transition_result['valid'] = False
            
            # Validate that the to_step input requirements are met
            to_input_validation = await self.validate_step_input(
                to_step, symbol, exchange, timeframe, data_dir
            )
            
            if not to_input_validation['valid']:
                transition_result['errors'].extend(to_input_validation['errors'])
                transition_result['valid'] = False
            
            # Check for data consistency between steps
            consistency_check = await self._check_data_consistency(
                from_step, to_step, symbol, exchange, timeframe, data_dir
            )
            
            if not consistency_check['valid']:
                transition_result['errors'].extend(consistency_check['errors'])
                transition_result['valid'] = False
            
            transition_result['transition_info'] = {
                'from_step_output': from_output_validation,
                'to_step_input': to_input_validation,
                'consistency_check': consistency_check,
            }
            
            if transition_result['valid']:
                self.logger.info(f"✅ Step transition validation passed: {from_step} -> {to_step}")
            else:
                self.logger.error(f"❌ Step transition validation failed: {from_step} -> {to_step}")
                self.logger.error(f"Errors: {transition_result['errors']}")
            
            return transition_result
            
        except Exception as e:
            self.logger.exception(f"❌ Step transition validation failed: {from_step} -> {to_step}: {e}")
            transition_result['valid'] = False
            transition_result['errors'].append(f"Transition validation exception: {str(e)}")
            return transition_result

    @handles_errors(Exception, fallback=False)
    async def _check_data_consistency(
        self,
        from_step: str,
        to_step: str,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
    ) -> Dict[str, Any]:
        """Check data consistency between steps."""
        consistency_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
        }
        
        try:
            # This is a simplified consistency check
            # In a real implementation, you would check things like:
            # - Row counts match between steps
            # - Time ranges are consistent
            # - Regime labels are preserved
            # - Feature dimensions are correct
            
            # For now, we'll just return a successful result
            # This can be enhanced based on specific requirements
            
            return consistency_result
            
        except Exception as e:
            self.logger.exception(f"❌ Data consistency check failed: {e}")
            consistency_result['valid'] = False
            consistency_result['errors'].append(f"Consistency check exception: {str(e)}")
            return consistency_result

    def get_step_schema(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a specific step."""
        return self.step_schemas.get(step_name)

    def get_all_step_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all step schemas."""
        return self.step_schemas.copy()

    def add_step_schema(self, step_name: str, schema: Dict[str, Any]) -> None:
        """Add or update a step schema."""
        self.step_schemas[step_name] = schema

    def remove_step_schema(self, step_name: str) -> bool:
        """Remove a step schema."""
        if step_name in self.step_schemas:
            del self.step_schemas[step_name]
            return True
        return False


# Global instance for easy access
enhanced_step_validator = EnhancedStepValidator()


# Convenience functions
async def validate_step_input(
    step_name: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
) -> Dict[str, Any]:
    """Validate input for a specific step."""
    return await enhanced_step_validator.validate_step_input(
        step_name, symbol, exchange, timeframe, data_dir
    )


async def validate_step_output(
    step_name: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
) -> Dict[str, Any]:
    """Validate output for a specific step."""
    return await enhanced_step_validator.validate_step_output(
        step_name, symbol, exchange, timeframe, data_dir
    )


async def validate_step_transition(
    from_step: str,
    to_step: str,
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
) -> Dict[str, Any]:
    """Validate transition between steps."""
    return await enhanced_step_validator.validate_step_transition(
        from_step, to_step, symbol, exchange, timeframe, data_dir
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        validator = EnhancedStepValidator()
        
        # Validate a step
        result = await validator.validate_step_input(
            step_name="hmm_clustering",
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache"
        )
        
        print(f"Validation result: {result}")
    
    asyncio.run(main())