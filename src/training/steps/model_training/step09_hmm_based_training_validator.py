"""HMM-Based Training Step Validator"""

import json
from typing import Any, Dict, List
import pandas as pd
from pathlib import Path

from src.core.decorators import handles_errors, validates, log_call, traced
from src.utils.logger import system_logger
from src.utils.common_operations import safe_file_exists, validate_dataframe_schema, validate_data_quality
from src.core.decorators.errors import handles_errors

class HMMTrainingValidator:
    """Validator for HMM-based training step."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('HMMTrainingValidator')
        self.validation_results: Dict[str, Any] = {}

    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def validate_training_step(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> Dict[str, Any]:
        """Validate HMM-based training step."""
        self.logger.info(f"🔍 Validating HMM-based training step for {symbol} on {exchange}")
        
        try:
            validation_result = {
                'step_name': 'step09_hmm_based_training',
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'overall_success': False,
                'errors': [],
                'warnings': []
            }
            
            # Validate inputs
            input_validation = await self._validate_inputs(symbol, exchange, timeframe, data_dir)
            if not input_validation['success']:
                validation_result['errors'].extend(input_validation['errors'])
                return validation_result
            
            # Validate configuration
            config_validation = await self._validate_configuration()
            if not config_validation['success']:
                validation_result['errors'].extend(config_validation['errors'])
                return validation_result
            
            # Validate data availability
            data_validation = await self._validate_data_availability(symbol, exchange, data_dir)
            if not data_validation['success']:
                validation_result['errors'].extend(data_validation['errors'])
                return validation_result
            
            validation_result['overall_success'] = True
            validation_result['warnings'].extend(input_validation.get('warnings', []))
            validation_result['warnings'].extend(config_validation.get('warnings', []))
            validation_result['warnings'].extend(data_validation.get('warnings', []))
            
            self.validation_results['step09_hmm_based_training'] = validation_result
            
            self.logger.info("✅ HMM-based training step validation completed successfully")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ HMM-based training step validation failed: {e}")
            validation_result['errors'].append(f"Validation error: {e}")
            return validation_result

    @handles_errors(Exception, fallback={'success': False, 'errors': ['Input validation failed']}, log_level="ERROR")
    async def _validate_inputs(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Dict[str, Any]:
        """Validate input parameters."""
        errors = []
        warnings = []
        
        if not symbol or not isinstance(symbol, str):
            errors.append("Symbol must be a non-empty string")
        elif len(symbol) < 3:
            errors.append("Symbol must be at least 3 characters long")
        elif not symbol.isupper():
            warnings.append("Symbol should be uppercase")
        
        valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
        if not exchange or exchange not in valid_exchanges:
            errors.append(f"Exchange must be one of: {valid_exchanges}")
        
        valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        if not timeframe or timeframe not in valid_timeframes:
            errors.append(f"Timeframe must be one of: {valid_timeframes}")
        
        if not data_dir or not isinstance(data_dir, str):
            errors.append("Data directory must be a non-empty string")
        elif not safe_file_exists(data_dir):
            errors.append(f"Data directory does not exist: {data_dir}")
        
        return {
            'success': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    @handles_errors(Exception, fallback={'success': False, 'errors': ['Configuration validation failed']}, log_level="ERROR")
    async def _validate_configuration(self) -> Dict[str, Any]:
        """Validate training configuration."""
        errors = []
        warnings = []
        
        required_sections = ['HMM_LM', 'regime_specific_training']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required configuration section: {section}")
        
        if 'regime_specific_training' in self.config:
            regime_config = self.config['regime_specific_training']
            if not isinstance(regime_config, dict):
                errors.append("Regime-specific training configuration must be a dictionary")
            else:
                required_regime_params = ['min_regime_samples', 'regime_validation_split']
                for param in required_regime_params:
                    if param not in regime_config:
                        errors.append(f"Missing required regime parameter: {param}")
        
        return {
            'success': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    @handles_errors(Exception, fallback={'success': False, 'errors': ['Data validation failed']}, log_level="ERROR")
    async def _validate_data_availability(self, symbol: str, exchange: str, data_dir: str) -> Dict[str, Any]:
        """Validate data availability and quality."""
        errors = []
        warnings = []
        
        required_files = [
            f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        for file_name in required_files:
            file_path = f"{data_dir}/{file_name}"
            if not safe_file_exists(file_path):
                errors.append(f"Required data file not found: {file_path}")
            else:
                file_size = Path(file_path).stat().st_size
                if file_size == 0:
                    errors.append(f"Data file is empty: {file_path}")
                elif file_size < 1024:
                    warnings.append(f"Data file is very small: {file_path} ({file_size} bytes)")
        
        main_data_file = f"{data_dir}/{required_files[0]}"
        if safe_file_exists(main_data_file):
            try:
                df = pd.read_parquet(main_data_file)
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                schema_valid, schema_errors = validate_dataframe_schema(df, required_columns)
                if not schema_valid:
                    errors.extend([f"Schema error: {error}" for error in schema_errors])
                
                if len(df) < 1000:
                    warnings.append(f"Low data volume: {len(df)} rows (minimum recommended: 1000)")
                    
            except Exception as e:
                errors.append(f"Failed to validate data file {main_data_file}: {e}")
        
        return {
            'success': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }