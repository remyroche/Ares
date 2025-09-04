#!/usr/bin/env python3
"""Comprehensive Optimisation Pipeline Validator.

This module provides comprehensive validation for the optimisation pipeline including:
- Data quality validation
- Step dependency validation
- Configuration validation
- Output validation
- Performance monitoring
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load
)
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates, traced, log_execution_time
from src.utils.base_validator import BaseValidator

logger = system_logger.getChild('OptimisationPipelineValidator')

class OptimisationPipelineValidator(BaseValidator):
    """Comprehensive validator for optimisation pipeline with enhanced data protection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("optimisation_pipeline", config)
        self.dq_framework = DataQualityFramework()
        self.validation_results = {}
        self.performance_metrics = {}
        
    @validates()
    async def validate_input_data_quality(self, symbol: str, exchange: str, data_dir: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate input data quality with comprehensive checks."""
        self.logger.info("🔍 Validating input data quality...")
        
        validation_metrics = {
            'files_checked': 0,
            'files_valid': 0,
            'data_quality_issues': [],
            'critical_issues': 0,
            'warnings': 0
        }
        
        try:
            # Check required data files
            required_files = [
                f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
                f"{data_dir}/volume_{exchange}_{symbol}_consolidated.parquet"
            ]
            
            for file_path in required_files:
                validation_metrics['files_checked'] += 1
                
                if not safe_file_exists(file_path):
                    validation_metrics['data_quality_issues'].append({
                        'type': 'missing_file',
                        'file': file_path,
                        'severity': 'critical'
                    })
                    validation_metrics['critical_issues'] += 1
                    continue
                
                # Load and validate data
                try:
                    df = pd.read_parquet(file_path)
                    
                    if df.empty:
                        validation_metrics['data_quality_issues'].append({
                            'type': 'empty_data',
                            'file': file_path,
                            'severity': 'critical'
                        })
                        validation_metrics['critical_issues'] += 1
                        continue
                    
                    # Comprehensive data quality validation
                    quality_result = self.dq_framework.validate_data(df, ['klines_schema'])
                    
                    if not quality_result.get('overall_passed', False):
                        for issue in quality_result.get('errors', []):
                            validation_metrics['data_quality_issues'].append({
                                'type': 'data_quality',
                                'file': file_path,
                                'issue': issue,
                                'severity': 'high'
                            })
                            validation_metrics['warnings'] += 1
                    
                    # Check for data completeness
                    completeness_score = self._calculate_data_completeness(df)
                    if completeness_score < 0.95:  # 95% completeness threshold
                        validation_metrics['data_quality_issues'].append({
                            'type': 'incomplete_data',
                            'file': file_path,
                            'completeness_score': completeness_score,
                            'severity': 'medium'
                        })
                        validation_metrics['warnings'] += 1
                    
                    validation_metrics['files_valid'] += 1
                    
                except Exception as e:
                    validation_metrics['data_quality_issues'].append({
                        'type': 'file_read_error',
                        'file': file_path,
                        'error': str(e),
                        'severity': 'critical'
                    })
                    validation_metrics['critical_issues'] += 1
            
            # Determine overall validation result
            validation_passed = validation_metrics['critical_issues'] == 0
            
            if validation_passed:
                self.logger.info(f"✅ Data quality validation passed: {validation_metrics['files_valid']}/{validation_metrics['files_checked']} files valid")
            else:
                self.logger.error(f"❌ Data quality validation failed: {validation_metrics['critical_issues']} critical issues")
            
            return validation_passed, validation_metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Data quality validation failed with exception: {e}")
            return False, {'error': str(e)}
    
    def _calculate_data_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        try:
            total_cells = df.size
            non_null_cells = df.count().sum()
            return non_null_cells / total_cells if total_cells > 0 else 0.0
        except Exception:
            return 0.0
    
    @validates()
    async def validate_step_dependencies(self, symbol: str, exchange: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate that all required step dependencies are met."""
        self.logger.info("🔍 Validating step dependencies...")
        
        dependency_metrics = {
            'dependencies_checked': 0,
            'dependencies_met': 0,
            'missing_dependencies': [],
            'warnings': []
        }
        
        try:
            # Define required dependencies
            required_dependencies = [
                {
                    'name': 'tactician_specialist_model',
                    'file': f"models/{symbol}_{exchange}_tactician_specialist.pkl",
                    'critical': True
                },
                {
                    'name': 'analyst_ensemble_model',
                    'file': f"models/{symbol}_{exchange}_analyst_ensemble.pkl",
                    'critical': False
                },
                {
                    'name': 'regime_classification_model',
                    'file': f"models/{symbol}_{exchange}_regime_classifier.pkl",
                    'critical': False
                },
                {
                    'name': 'feature_engineering_config',
                    'file': f"config/{symbol}_{exchange}_feature_config.json",
                    'critical': False
                }
            ]
            
            for dep in required_dependencies:
                dependency_metrics['dependencies_checked'] += 1
                
                if safe_file_exists(dep['file']):
                    dependency_metrics['dependencies_met'] += 1
                    self.logger.info(f"✅ Dependency met: {dep['name']}")
                else:
                    missing_dep = {
                        'name': dep['name'],
                        'file': dep['file'],
                        'critical': dep['critical']
                    }
                    dependency_metrics['missing_dependencies'].append(missing_dep)
                    
                    if dep['critical']:
                        self.logger.error(f"❌ Critical dependency missing: {dep['name']}")
                    else:
                        self.logger.warning(f"⚠️ Optional dependency missing: {dep['name']}")
                        dependency_metrics['warnings'].append(f"Optional dependency missing: {dep['name']}")
            
            # Check for critical dependencies
            critical_missing = [dep for dep in dependency_metrics['missing_dependencies'] if dep['critical']]
            validation_passed = len(critical_missing) == 0
            
            if validation_passed:
                self.logger.info(f"✅ Step dependencies validation passed: {dependency_metrics['dependencies_met']}/{dependency_metrics['dependencies_checked']} dependencies met")
            else:
                self.logger.error(f"❌ Step dependencies validation failed: {len(critical_missing)} critical dependencies missing")
            
            return validation_passed, dependency_metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Step dependencies validation failed with exception: {e}")
            return False, {'error': str(e)}
    
    @validates()
    async def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate optimisation configuration parameters."""
        self.logger.info("🔍 Validating configuration...")
        
        config_metrics = {
            'parameters_checked': 0,
            'parameters_valid': 0,
            'invalid_parameters': [],
            'warnings': []
        }
        
        try:
            # Define configuration validation rules
            config_rules = {
                'confidence_calibration': {'type': bool, 'required': True},
                'parameter_optimization': {'type': bool, 'required': True},
                'random_state': {'type': int, 'required': True, 'min': 0, 'max': 2**32-1},
                'force_rerun': {'type': bool, 'required': False, 'default': False},
                'enhanced_mode': {'type': bool, 'required': False, 'default': False},
                'data_quality_checks': {'type': bool, 'required': False, 'default': True},
                'comprehensive_logging': {'type': bool, 'required': False, 'default': True},
                'validation_enabled': {'type': bool, 'required': False, 'default': True}
            }
            
            for param_name, rules in config_rules.items():
                config_metrics['parameters_checked'] += 1
                
                if param_name not in config:
                    if rules.get('required', False):
                        config_metrics['invalid_parameters'].append({
                            'parameter': param_name,
                            'issue': 'missing_required_parameter',
                            'severity': 'critical'
                        })
                        continue
                    else:
                        # Use default value
                        config[param_name] = rules.get('default', None)
                        config_metrics['warnings'].append(f"Using default value for {param_name}: {config[param_name]}")
                
                param_value = config[param_name]
                
                # Type validation
                if not isinstance(param_value, rules['type']):
                    config_metrics['invalid_parameters'].append({
                        'parameter': param_name,
                        'issue': 'type_mismatch',
                        'expected_type': rules['type'].__name__,
                        'actual_type': type(param_value).__name__,
                        'severity': 'critical'
                    })
                    continue
                
                # Range validation for numeric types
                if rules['type'] == int and 'min' in rules:
                    if param_value < rules['min'] or param_value > rules.get('max', float('inf')):
                        config_metrics['invalid_parameters'].append({
                            'parameter': param_name,
                            'issue': 'value_out_of_range',
                            'value': param_value,
                            'min': rules['min'],
                            'max': rules.get('max'),
                            'severity': 'critical'
                        })
                        continue
                
                config_metrics['parameters_valid'] += 1
            
            validation_passed = len(config_metrics['invalid_parameters']) == 0
            
            if validation_passed:
                self.logger.info(f"✅ Configuration validation passed: {config_metrics['parameters_valid']}/{config_metrics['parameters_checked']} parameters valid")
            else:
                self.logger.error(f"❌ Configuration validation failed: {len(config_metrics['invalid_parameters'])} invalid parameters")
            
            return validation_passed, config_metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Configuration validation failed with exception: {e}")
            return False, {'error': str(e)}
    
    @validates()
    async def validate_output_quality(self, symbol: str, exchange: str, expected_outputs: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """Validate output quality after optimisation."""
        self.logger.info("🔍 Validating output quality...")
        
        output_metrics = {
            'outputs_checked': 0,
            'outputs_valid': 0,
            'missing_outputs': [],
            'invalid_outputs': [],
            'warnings': []
        }
        
        try:
            for output_file in expected_outputs:
                output_metrics['outputs_checked'] += 1
                
                if not safe_file_exists(output_file):
                    output_metrics['missing_outputs'].append(output_file)
                    self.logger.error(f"❌ Expected output missing: {output_file}")
                    continue
                
                # Validate output file content
                try:
                    if output_file.endswith('.json'):
                        data = safe_json_load(output_file)
                        if not isinstance(data, dict) or not data:
                            output_metrics['invalid_outputs'].append({
                                'file': output_file,
                                'issue': 'invalid_json_content',
                                'severity': 'high'
                            })
                            continue
                    
                    elif output_file.endswith('.pkl'):
                        # Basic pickle file validation
                        import pickle
                        with open(output_file, 'rb') as f:
                            data = pickle.load(f)
                        if data is None:
                            output_metrics['invalid_outputs'].append({
                                'file': output_file,
                                'issue': 'empty_pickle_content',
                                'severity': 'high'
                            })
                            continue
                    
                    output_metrics['outputs_valid'] += 1
                    self.logger.info(f"✅ Output valid: {output_file}")
                    
                except Exception as e:
                    output_metrics['invalid_outputs'].append({
                        'file': output_file,
                        'issue': 'file_read_error',
                        'error': str(e),
                        'severity': 'high'
                    })
            
            validation_passed = len(output_metrics['missing_outputs']) == 0 and len(output_metrics['invalid_outputs']) == 0
            
            if validation_passed:
                self.logger.info(f"✅ Output quality validation passed: {output_metrics['outputs_valid']}/{output_metrics['outputs_checked']} outputs valid")
            else:
                self.logger.error(f"❌ Output quality validation failed: {len(output_metrics['missing_outputs'])} missing, {len(output_metrics['invalid_outputs'])} invalid")
            
            return validation_passed, output_metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Output quality validation failed with exception: {e}")
            return False, {'error': str(e)}
    
    @traced(span_name="comprehensive_pipeline_validation")
    @log_execution_time("pipeline_validation")
    async def validate_comprehensive_pipeline(
        self, 
        symbol: str, 
        exchange: str, 
        data_dir: str, 
        config: Dict[str, Any],
        expected_outputs: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform comprehensive pipeline validation."""
        self.logger.info("🚀 Starting comprehensive pipeline validation...")
        
        validation_start_time = get_current_datetime()
        comprehensive_results = {
            'validation_start_time': format_datetime(validation_start_time, '%Y-%m-%d %H:%M:%S'),
            'validation_results': {},
            'overall_passed': False,
            'critical_issues': 0,
            'warnings': 0
        }
        
        try:
            # Run all validation steps
            validation_tasks = [
                self.validate_input_data_quality(symbol, exchange, data_dir),
                self.validate_step_dependencies(symbol, exchange),
                self.validate_configuration(config)
            ]
            
            # Add output validation if expected outputs provided
            if expected_outputs:
                validation_tasks.append(self.validate_output_quality(symbol, exchange, expected_outputs))
            
            # Execute all validations
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            validation_names = [
                'input_data_quality',
                'step_dependencies', 
                'configuration'
            ]
            if expected_outputs:
                validation_names.append('output_quality')
            
            for i, (name, result) in enumerate(zip(validation_names, validation_results)):
                if isinstance(result, Exception):
                    comprehensive_results['validation_results'][name] = {
                        'passed': False,
                        'error': str(result),
                        'severity': 'critical'
                    }
                    comprehensive_results['critical_issues'] += 1
                else:
                    passed, metrics = result
                    comprehensive_results['validation_results'][name] = {
                        'passed': passed,
                        'metrics': metrics
                    }
                    if not passed:
                        comprehensive_results['critical_issues'] += 1
                    comprehensive_results['warnings'] += metrics.get('warnings', 0)
            
            # Determine overall result
            comprehensive_results['overall_passed'] = comprehensive_results['critical_issues'] == 0
            
            validation_end_time = get_current_datetime()
            comprehensive_results['validation_end_time'] = format_datetime(validation_end_time, '%Y-%m-%d %H:%M:%S')
            comprehensive_results['validation_duration'] = (validation_end_time - validation_start_time).total_seconds()
            
            if comprehensive_results['overall_passed']:
                self.logger.info("🎉 Comprehensive pipeline validation passed!")
            else:
                self.logger.error(f"❌ Comprehensive pipeline validation failed: {comprehensive_results['critical_issues']} critical issues")
            
            return comprehensive_results['overall_passed'], comprehensive_results
            
        except Exception as e:
            self.logger.exception(f"❌ Comprehensive pipeline validation failed with exception: {e}")
            comprehensive_results['error'] = str(e)
            return False, comprehensive_results