#!/usr/bin/env python3
"""
Market Analysis Pipeline Validators

This module provides comprehensive validators for each step of the market analysis pipeline:
1. Data collection validation
2. HMM clustering validation
3. Feature engineering validation
4. Data quality validation
5. Pipeline integrity validation
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common_operations import (
    format_datetime,
    get_current_datetime,
    safe_file_exists,
    safe_json_load,
)
from src.utils.base_validator import BaseValidator
from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.security_framework import SecurityFramework

logger = system_logger.getChild("MarketAnalysisValidators")


class DataCollectionValidator(BaseValidator):
    """Validator for data collection step."""
    
    def __init__(self):
        super().__init__()
        self.validator_name = "data_collection_validator"
        self.logger = system_logger.getChild("DataCollectionValidator")
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data collection step."""
        start_time = time.time()
        self.logger.info("🔍 Validating data collection step...")
        
        try:
            validation_results = {
                'validation_passed': True,
                'warnings': [],
                'errors': [],
                'details': {},
                'timestamp': get_current_datetime().isoformat()
            }
            
            # Extract parameters
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            # Validate symbol format
            symbol_validation = await self._validate_symbol(symbol)
            if not symbol_validation['valid']:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(symbol_validation['error'])
            else:
                validation_results['details']['symbol_validation'] = symbol_validation
            
            # Validate exchange
            exchange_validation = await self._validate_exchange(exchange)
            if not exchange_validation['valid']:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(exchange_validation['error'])
            else:
                validation_results['details']['exchange_validation'] = exchange_validation
            
            # Validate timeframe
            timeframe_validation = await self._validate_timeframe(timeframe)
            if not timeframe_validation['valid']:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(timeframe_validation['error'])
            else:
                validation_results['details']['timeframe_validation'] = timeframe_validation
            
            # Validate data directory
            data_dir_validation = await self._validate_data_directory(data_dir)
            if not data_dir_validation['valid']:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(data_dir_validation['error'])
            else:
                validation_results['details']['data_dir_validation'] = data_dir_validation
            
            # Check if data already exists
            data_file = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
            if safe_file_exists(data_file):
                validation_results['details']['data_exists'] = True
                validation_results['details']['data_file'] = str(data_file)
            else:
                validation_results['details']['data_exists'] = False
                validation_results['warnings'].append(f"Data file not found: {data_file}")
            
            # Validate configuration
            config_validation = await self._validate_data_collection_config(config)
            if not config_validation['valid']:
                validation_results['warnings'].extend(config_validation['warnings'])
            
            validation_results['details']['config_validation'] = config_validation
            
            duration = time.time() - start_time
            validation_results['duration'] = duration
            
            if validation_results['validation_passed']:
                self.logger.info(f"✅ Data collection validation passed in {duration:.3f}s")
            else:
                self.logger.error(f"❌ Data collection validation failed in {duration:.3f}s")
            
            return validation_results
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Data collection validation failed with exception: {e}")
            return {
                'validation_passed': False,
                'error': str(e),
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _validate_symbol(self, symbol: str) -> Dict[str, Any]:
        """Validate trading symbol."""
        try:
            if not symbol or not isinstance(symbol, str):
                return {'valid': False, 'error': 'Symbol must be a non-empty string'}
            
            if len(symbol) < 3:
                return {'valid': False, 'error': 'Symbol must be at least 3 characters long'}
            
            # Check if symbol contains valid characters
            if not symbol.replace('USDT', '').replace('BTC', '').replace('ETH', '').isalnum():
                return {'valid': False, 'error': 'Symbol contains invalid characters'}
            
            return {'valid': True, 'symbol': symbol}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_exchange(self, exchange: str) -> Dict[str, Any]:
        """Validate exchange name."""
        try:
            valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
            
            if not exchange or not isinstance(exchange, str):
                return {'valid': False, 'error': 'Exchange must be a non-empty string'}
            
            if exchange.upper() not in valid_exchanges:
                return {'valid': False, 'error': f'Exchange must be one of: {valid_exchanges}'}
            
            return {'valid': True, 'exchange': exchange.upper()}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """Validate timeframe."""
        try:
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            
            if not timeframe or not isinstance(timeframe, str):
                return {'valid': False, 'error': 'Timeframe must be a non-empty string'}
            
            if timeframe not in valid_timeframes:
                return {'valid': False, 'error': f'Timeframe must be one of: {valid_timeframes}'}
            
            return {'valid': True, 'timeframe': timeframe}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_data_directory(self, data_dir: str) -> Dict[str, Any]:
        """Validate data directory."""
        try:
            if not data_dir or not isinstance(data_dir, str):
                return {'valid': False, 'error': 'Data directory must be a non-empty string'}
            
            data_path = Path(data_dir)
            
            # Check if directory exists or can be created
            if not data_path.exists():
                try:
                    data_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return {'valid': False, 'error': f'Cannot create data directory: {e}'}
            
            # Check if directory is writable
            if not data_path.is_dir():
                return {'valid': False, 'error': 'Data directory path is not a directory'}
            
            return {'valid': True, 'data_dir': str(data_path)}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_data_collection_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data collection configuration."""
        try:
            warnings = []
            
            # Check for required configuration keys
            required_keys = ['symbol', 'exchange', 'timeframe']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                warnings.append(f"Missing configuration keys: {missing_keys}")
            
            # Validate lookback period
            lookback_days = config.get('lookback_days', 30)
            if not isinstance(lookback_days, int) or lookback_days <= 0:
                warnings.append("lookback_days must be a positive integer")
            
            # Validate force_rerun flag
            force_rerun = config.get('force_rerun', False)
            if not isinstance(force_rerun, bool):
                warnings.append("force_rerun must be a boolean")
            
            return {'valid': len(warnings) == 0, 'warnings': warnings}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}


class HMMClusteringValidator(BaseValidator):
    """Validator for HMM clustering step."""
    
    def __init__(self):
        super().__init__()
        self.validator_name = "hmm_clustering_validator"
        self.logger = system_logger.getChild("HMMClusteringValidator")
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HMM clustering step."""
        start_time = time.time()
        self.logger.info("🔍 Validating HMM clustering step...")
        
        try:
            validation_results = {
                'validation_passed': True,
                'warnings': [],
                'errors': [],
                'details': {},
                'timestamp': get_current_datetime().isoformat()
            }
            
            # Extract parameters
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            # Validate data availability
            data_validation = await self._validate_data_availability(symbol, exchange, data_dir)
            if not data_validation['valid']:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(data_validation['error'])
            else:
                validation_results['details']['data_validation'] = data_validation
            
            # Validate HMM configuration
            hmm_config_validation = await self._validate_hmm_config(config)
            if not hmm_config_validation['valid']:
                validation_results['warnings'].extend(hmm_config_validation['warnings'])
            
            validation_results['details']['hmm_config_validation'] = hmm_config_validation
            
            # Validate previous step results
            if 'data_collection' in pipeline_state:
                prev_step_validation = await self._validate_previous_step_results(
                    pipeline_state['data_collection']
                )
                if not prev_step_validation['valid']:
                    validation_results['warnings'].extend(prev_step_validation['warnings'])
                
                validation_results['details']['prev_step_validation'] = prev_step_validation
            
            duration = time.time() - start_time
            validation_results['duration'] = duration
            
            if validation_results['validation_passed']:
                self.logger.info(f"✅ HMM clustering validation passed in {duration:.3f}s")
            else:
                self.logger.error(f"❌ HMM clustering validation failed in {duration:.3f}s")
            
            return validation_results
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ HMM clustering validation failed with exception: {e}")
            return {
                'validation_passed': False,
                'error': str(e),
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _validate_data_availability(self, symbol: str, exchange: str, data_dir: str) -> Dict[str, Any]:
        """Validate that required data is available for HMM clustering."""
        try:
            data_file = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
            
            if not safe_file_exists(data_file):
                return {
                    'valid': False,
                    'error': f'Required data file not found: {data_file}'
                }
            
            # Check file size
            file_size = data_file.stat().st_size
            if file_size < 1024:  # Less than 1KB
                return {
                    'valid': False,
                    'error': f'Data file is too small: {file_size} bytes'
                }
            
            return {
                'valid': True,
                'data_file': str(data_file),
                'file_size': file_size
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_hmm_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HMM clustering configuration."""
        try:
            warnings = []
            
            # Validate number of states
            n_states = config.get('n_states', 3)
            if not isinstance(n_states, int) or n_states < 2 or n_states > 10:
                warnings.append("n_states should be an integer between 2 and 10")
            
            # Validate random state
            random_state = config.get('random_state', 42)
            if not isinstance(random_state, int):
                warnings.append("random_state must be an integer")
            
            # Validate optimization parameters
            n_trials = config.get('n_trials', 50)
            if not isinstance(n_trials, int) or n_trials < 10:
                warnings.append("n_trials should be an integer >= 10")
            
            return {'valid': len(warnings) == 0, 'warnings': warnings}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_previous_step_results(self, prev_step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate previous step results."""
        try:
            warnings = []
            
            if not prev_step_result.get('success', False):
                warnings.append("Previous step (data collection) did not complete successfully")
            
            if not prev_step_result.get('data_exists', False):
                warnings.append("Previous step did not produce required data")
            
            return {'valid': len(warnings) == 0, 'warnings': warnings}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}


class FeatureEngineeringValidator(BaseValidator):
    """Validator for feature engineering step."""
    
    def __init__(self):
        super().__init__()
        self.validator_name = "feature_engineering_validator"
        self.logger = system_logger.getChild("FeatureEngineeringValidator")
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature engineering step."""
        start_time = time.time()
        self.logger.info("🔍 Validating feature engineering step...")
        
        try:
            validation_results = {
                'validation_passed': True,
                'warnings': [],
                'errors': [],
                'details': {},
                'timestamp': get_current_datetime().isoformat()
            }
            
            # Extract parameters
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data_cache')
            
            # Validate data availability
            data_validation = await self._validate_data_availability(symbol, exchange, data_dir)
            if not data_validation['valid']:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(data_validation['error'])
            else:
                validation_results['details']['data_validation'] = data_validation
            
            # Validate feature engineering configuration
            feature_config_validation = await self._validate_feature_config(config)
            if not feature_config_validation['valid']:
                validation_results['warnings'].extend(feature_config_validation['warnings'])
            
            validation_results['details']['feature_config_validation'] = feature_config_validation
            
            # Validate previous step results
            if 'hmm_clustering' in pipeline_state:
                prev_step_validation = await self._validate_previous_step_results(
                    pipeline_state['hmm_clustering']
                )
                if not prev_step_validation['valid']:
                    validation_results['warnings'].extend(prev_step_validation['warnings'])
                
                validation_results['details']['prev_step_validation'] = prev_step_validation
            
            duration = time.time() - start_time
            validation_results['duration'] = duration
            
            if validation_results['validation_passed']:
                self.logger.info(f"✅ Feature engineering validation passed in {duration:.3f}s")
            else:
                self.logger.error(f"❌ Feature engineering validation failed in {duration:.3f}s")
            
            return validation_results
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Feature engineering validation failed with exception: {e}")
            return {
                'validation_passed': False,
                'error': str(e),
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _validate_data_availability(self, symbol: str, exchange: str, data_dir: str) -> Dict[str, Any]:
        """Validate that required data is available for feature engineering."""
        try:
            data_file = Path(data_dir) / f"aggtrades_{exchange}_{symbol}_consolidated.parquet"
            
            if not safe_file_exists(data_file):
                return {
                    'valid': False,
                    'error': f'Required data file not found: {data_file}'
                }
            
            return {
                'valid': True,
                'data_file': str(data_file)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_feature_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature engineering configuration."""
        try:
            warnings = []
            
            # Validate feature types
            feature_types = config.get('feature_types', ['technical', 'statistical'])
            if not isinstance(feature_types, list):
                warnings.append("feature_types must be a list")
            
            # Validate window sizes
            window_sizes = config.get('window_sizes', [5, 10, 20, 50])
            if not isinstance(window_sizes, list) or not all(isinstance(w, int) and w > 0 for w in window_sizes):
                warnings.append("window_sizes must be a list of positive integers")
            
            # Validate random state
            random_state = config.get('random_state', 42)
            if not isinstance(random_state, int):
                warnings.append("random_state must be an integer")
            
            return {'valid': len(warnings) == 0, 'warnings': warnings}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_previous_step_results(self, prev_step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate previous step results."""
        try:
            warnings = []
            
            if not prev_step_result.get('success', False):
                warnings.append("Previous step (HMM clustering) did not complete successfully")
            
            if not prev_step_result.get('regime_model'):
                warnings.append("Previous step did not produce regime model")
            
            return {'valid': len(warnings) == 0, 'warnings': warnings}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}


class PipelineIntegrityValidator(BaseValidator):
    """Validator for overall pipeline integrity."""
    
    def __init__(self):
        super().__init__()
        self.validator_name = "pipeline_integrity_validator"
        self.logger = system_logger.getChild("PipelineIntegrityValidator")
    
    async def validate(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall pipeline integrity."""
        start_time = time.time()
        self.logger.info("🔍 Validating pipeline integrity...")
        
        try:
            validation_results = {
                'validation_passed': True,
                'warnings': [],
                'errors': [],
                'details': {},
                'timestamp': get_current_datetime().isoformat()
            }
            
            # Validate training input consistency
            input_consistency = await self._validate_input_consistency(training_input)
            if not input_consistency['valid']:
                validation_results['validation_passed'] = False
                validation_results['errors'].append(input_consistency['error'])
            else:
                validation_results['details']['input_consistency'] = input_consistency
            
            # Validate pipeline state consistency
            state_consistency = await self._validate_state_consistency(pipeline_state)
            if not state_consistency['valid']:
                validation_results['warnings'].extend(state_consistency['warnings'])
            
            validation_results['details']['state_consistency'] = state_consistency
            
            # Validate configuration consistency
            config_consistency = await self._validate_config_consistency(config)
            if not config_consistency['valid']:
                validation_results['warnings'].extend(config_consistency['warnings'])
            
            validation_results['details']['config_consistency'] = config_consistency
            
            duration = time.time() - start_time
            validation_results['duration'] = duration
            
            if validation_results['validation_passed']:
                self.logger.info(f"✅ Pipeline integrity validation passed in {duration:.3f}s")
            else:
                self.logger.error(f"❌ Pipeline integrity validation failed in {duration:.3f}s")
            
            return validation_results
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(f"❌ Pipeline integrity validation failed with exception: {e}")
            return {
                'validation_passed': False,
                'error': str(e),
                'duration': duration,
                'timestamp': get_current_datetime().isoformat()
            }
    
    async def _validate_input_consistency(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training input consistency."""
        try:
            # Check for required keys
            required_keys = ['symbol', 'exchange', 'timeframe', 'data_dir']
            missing_keys = [key for key in required_keys if key not in training_input]
            
            if missing_keys:
                return {
                    'valid': False,
                    'error': f'Missing required training input keys: {missing_keys}'
                }
            
            # Validate data types
            if not isinstance(training_input['symbol'], str):
                return {'valid': False, 'error': 'Symbol must be a string'}
            
            if not isinstance(training_input['exchange'], str):
                return {'valid': False, 'error': 'Exchange must be a string'}
            
            if not isinstance(training_input['timeframe'], str):
                return {'valid': False, 'error': 'Timeframe must be a string'}
            
            if not isinstance(training_input['data_dir'], str):
                return {'valid': False, 'error': 'Data directory must be a string'}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_state_consistency(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline state consistency."""
        try:
            warnings = []
            
            # Check for expected steps
            expected_steps = ['data_collection', 'hmm_clustering', 'feature_engineering']
            present_steps = [step for step in expected_steps if step in pipeline_state]
            
            if len(present_steps) != len(expected_steps):
                warnings.append(f"Expected {len(expected_steps)} steps, found {len(present_steps)}")
            
            # Check step success status
            for step in present_steps:
                step_result = pipeline_state[step]
                if not step_result.get('success', False):
                    warnings.append(f"Step {step} did not complete successfully")
            
            return {'valid': len(warnings) == 0, 'warnings': warnings}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    async def _validate_config_consistency(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration consistency."""
        try:
            warnings = []
            
            # Check for conflicting settings
            if config.get('force_rerun', False) and config.get('use_cached_data', False):
                warnings.append("force_rerun and use_cached_data are conflicting settings")
            
            # Check for reasonable parameter ranges
            lookback_days = config.get('lookback_days', 30)
            if lookback_days < 1 or lookback_days > 3650:  # 10 years
                warnings.append("lookback_days should be between 1 and 3650")
            
            return {'valid': len(warnings) == 0, 'warnings': warnings}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}


# Validator registry for easy access
VALIDATOR_REGISTRY = {
    'data_collection': DataCollectionValidator,
    'hmm_clustering': HMMClusteringValidator,
    'feature_engineering': FeatureEngineeringValidator,
    'pipeline_integrity': PipelineIntegrityValidator,
}


def get_validator(validator_name: str) -> BaseValidator:
    """Get validator instance by name."""
    if validator_name not in VALIDATOR_REGISTRY:
        raise ValueError(f"Unknown validator: {validator_name}")
    
    return VALIDATOR_REGISTRY[validator_name]()


async def run_validator(validator_name: str, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run validator by name."""
    validator = get_validator(validator_name)
    return await validator.validate(training_input, pipeline_state, config)