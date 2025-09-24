from src.utils.tprint import tprint

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import pandas as pd  # noqa: F401
from src.utils.logger import system_logger
from src.utils.data.klines_parquet import get_klines_manager

# Import our standardized validation utilities
from .validation_utils import get_validator, ValidationErrorType, ValidationResult, validate_training_input, validate_pipeline_state
from .config_utils import get_config_manager, get_path_manager

# Use existing data validation utilities
from src.utils.data.validation.validators import CrossStepValidator
from src.utils.data.quality.data_quality import DataQualityFramework

# Standardized imports from utils
from src.utils.core.common import (
    safe_read_parquet,
    safe_json_dump,
    safe_json_load,
)
from src.utils.common_operations import get_logger, safe_dict_get, safe_float, safe_int, optimize_dataframe_dtypes, safe_divide, safe_log, safe_sqrt, safe_kelly_calculation, validate_positive, validate_range, MathValidationError
from src.utils.parquet_utils import get_parquet_utils
# Core decorators imports
from src.core.decorators import (
    handles_errors,
    traced,
    validates,
    log_execution_time,
    cached,
    error_boundary,
    timeout,
    retry
)
# Core errors imports
from src.core.errors import (
    AppError,
    ValidationError,
    DataIntegrityError,
    NotFoundError,
    TimeoutError
)

"""Validator for Step 4: Regime Data Splitting.

This module validates the regime data splitting step outputs with support for 10+ regimes.
"""
import json
import pandas as pd
from src.core.decorators.logging import log_execution_time, log_call

def smart_validation_cache(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator

def validate_step4_comprehensive(*args, **kwargs):
    def decorator(func: Callable):
        return func
    return decorator
try:
    from src.utils.logger import system_logger
except ImportError:
    import logging
    import datetime
    system_logger = logging.getLogger(__name__)

# Note: safe_json_load already imported from src.utils.core.common

class BaseValidator:

    def __init__(self, step_name: str, config: dict) -> None:
        self.step_name = step_name
        self.config = config
        try:
            from src.utils.data.quality.data_quality import DataQualityFramework as EnhancedDataQualityValidator  # type: ignore
            self._dq_validator = EnhancedDataQualityValidator()
        except Exception:
            self._dq_validator = None

    def validate_file_exists(self, path: str, description: str='file') -> tuple[bool, dict[str, Any]]:
        """Check if a file exists and return simple metrics."""
        try:
            p = Path(path)
            if not p.exists():
                return False, {'path': str(p), 'description': description, 'exists': False}
            stat = p.stat()
            return True, {
                'path': str(p),
                'description': description,
                'exists': True,
                'size_bytes': stat.st_size,
                'modified_ts': stat.st_mtime,
            }
        except Exception as e:
            return False, {'path': path, 'description': description, 'exists': False, 'error': str(e)}

    def validate_dataframe_quality(self, df: Any, min_rows: int = 0, required_columns: Optional[List[str]]=None, check_data_types: bool = False, check_value_ranges: bool = False, check_duplicates: bool = False, check_temporal_consistency: bool = False) -> tuple[bool, dict[str, Any]]:
        """Validate DataFrame quality using shared validator if available, else basic checks."""
        metrics: dict[str, Any] = {}
        passed = True
        try:
            pass
        except Exception:
            return False, {'error': 'pandas_not_available'}
        if df is None:
            return False, {'error': 'none_dataframe'}
        try:
            metrics['rows'] = int(len(df))
            metrics['columns'] = list(getattr(df, 'columns', []))
        except Exception:
            pass
        if len(df) < int(min_rows):
            passed = False
            metrics['min_rows_failed'] = {'required': int(min_rows), 'actual': int(len(df))}
        if required_columns:
            missing = [c for c in required_columns if c not in getattr(df, 'columns', [])]
            metrics['required_columns_missing'] = missing
            if missing:
                passed = False
        if self._dq_validator is not None:
            try:
                result = self._dq_validator.validate_dataframe_quality(df, context = self.step_name)
                metrics['dq_summary'] = result.get_summary()
                passed = passed and bool(result.passed)
            except Exception as e:
                metrics['dq_error'] = str(e)
        # Light-weight optional checks toggles can be used for future expansion
        return passed, metrics
logger = system_logger.getChild('Step4RegimeDataSplittingValidator')

class Step4RegimeDataSplittingValidator(BaseValidator):
    """Validator for Step 4: Regime Data Splitting."""

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            super().__init__('step04_regime_data_splitting', config)
        except Exception:
            pass
        self.logger = system_logger.getChild('Validator.Step4')
        
        # Initialize data validation using existing utilities
        self.cross_step_validator = CrossStepValidator()
        self.data_quality_framework = DataQualityFramework()

    async def validate_step4_regime_data_splitting(self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]) -> bool:
        """Validate Step 4: Regime Data Splitting."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info('🔍 Starting Step 4: Regime Data Splitting validation')
        try:
            regime_splits_dir = Path(data_dir) / exchange.lower() / symbol.lower() / 'regime_splits'
            if not regime_splits_dir.exists():
                self.logger.warning(f'⚠️ Regime splits directory not found: {regime_splits_dir}')
                return False
            regime_files = list(regime_splits_dir.glob('*_unified_regime_data.parquet'))
            if not regime_files:
                self.logger.warning('⚠️ No regime split files found')
                return False
            for regime_file in regime_files:
                if not self._validate_regime_file(regime_file):
                    return False
            timeframe = training_input.get('timeframe', '1m') if isinstance(training_input, dict) else '1m'
            stats_file = Path(data_dir) / exchange.lower() / symbol.lower() / 'models' / 'regime_statistics.json'
            if not stats_file.exists():
                self.logger.warning(f'⚠️ Regime statistics file not found: {stats_file}')
                return False
            if not self._validate_statistics_file(stats_file):
                return False
            self.logger.info('✅ Step 4: Regime Data Splitting validation passed')
            return True
        except Exception as e:
            error_context = {'step': 'step04_regime_data_splitting', 'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir, 'error_type': type(e).__name__, 'error_message': str(e), 'timestamp': pd.Timestamp.now().isoformat()}
            self.logger.exception(f'❌ Step 4 validation failed: {error_context}')
            return False

    @smart_validation_cache(ttl_seconds = 300)
    def _validate_regime_file(self, regime_file: Path) -> bool:
        """Validate a regime split file with caching."""
        try:
            self.logger.info(f'📁 Validating regime file: {regime_file.name}')
            file_exists, file_metrics = self.validate_file_exists(str(regime_file), 'regime file')
            if not file_exists:
                return False
            # For processed data validation, use safe parquet reader
            df = safe_read_parquet(regime_file)

            df_valid, df_metrics = self.validate_dataframe_quality(df = df, min_rows = 100, required_columns=['timestamp', 'composite_cluster_id'], check_data_types = True, check_value_ranges = True, check_duplicates = True, check_temporal_consistency = True)
            if not df_valid:
                self.logger.warning(f'⚠️ DataFrame validation failed for {regime_file.name}')
                return False
            if 'composite_cluster_id' in df.columns:
                unique_regimes = df['composite_cluster_id'].nunique()
                
                # Use existing validation patterns for regime count
                if unique_regimes < 2:
                    self.logger.warning(f'⚠️ Very few regimes ({unique_regimes}) in {regime_file.name}')
                elif unique_regimes > 100:
                    self.logger.info(f'📊 Large number of regimes ({unique_regimes}) in {regime_file.name} - using optimized processing')
                else:
                    self.logger.info(f'📊 Standard regime count ({unique_regimes}) in {regime_file.name}')
            self.logger.info(f'✅ Regime file validated: {regime_file.name}')
            return True
        except Exception as e:
            error_context = {'file': str(regime_file), 'error_type': type(e).__name__, 'error_message': str(e)}
            self.logger.exception(f'❌ Failed to validate regime file: {error_context}')
            return False

    @smart_validation_cache(ttl_seconds = 600)
    def _validate_statistics_file(self, stats_file: Path) -> bool:
        """Validate the regime statistics file with caching."""
        try:
            self.logger.info(f'📊 Validating statistics file: {stats_file.name}')
            file_exists, file_metrics = self.validate_file_exists(str(stats_file), 'statistics file')
            if not file_exists:
                return False
            stats_data = safe_json_load(stats_file)
            if not isinstance(stats_data, dict):
                self.logger.warning('⚠️ Statistics file should contain a dictionary')
                return False
            if not stats_data:
                self.logger.warning('⚠️ Empty statistics data')
                return False
            for regime_id, stats in stats_data.items():
                if not isinstance(stats, dict):
                    self.logger.warning(f'⚠️ Invalid statistics format for regime {regime_id}')
                    return False
                expected_new = {'count', 'percentage', 'mean_volatility', 'mean_momentum'}
                expected_old = {'count', 'duration_minutes', 'mean_volume'}
                keys = set(stats.keys())
                if expected_new.issubset(keys):
                    continue
                if expected_old.issubset(keys):
                    # Accept older stats schema for backward compatibility
                    continue
                missing_new = list(expected_new - keys)
                missing_old = list(expected_old - keys)
                self.logger.warning(f"⚠️ Statistics for regime {regime_id} missing required fields. Missing (new schema): {missing_new}; Missing (old schema): {missing_old}")
                return False
            self.logger.info(f'✅ Statistics file validated: {stats_file.name}')
            return True
        except Exception as e:
            error_context = {'file': str(stats_file), 'error_type': type(e).__name__, 'error_message': str(e)}
            self.logger.exception(f'❌ Failed to validate statistics file: {error_context}')
            return False

    def validate_prerequisites(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate prerequisites for Step 4 using BaseValidator methods."""
        validation_result = {'validation_passed': True, 'warnings': [], 'errors': [], 'details': {}}
        try:
            step03_output_dir = Path('generated/market_analysis/hmm_regimes')
            step03_files = list(step03_output_dir.glob(f'{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'))
            if not step03_files:
                validation_result['validation_passed'] = False
                validation_result['errors'].append(f'Step 03 HMM regime discovery output not found for {exchange}_{symbol}_{timeframe}')
            else:
                for file_path in step03_files:
                    try:
                        file_valid, file_metrics = self.validate_file_exists(str(file_path), 'step03 output file')
                    except Exception:
                        file_valid = file_path.exists()
                    if not file_valid:
                        validation_result['warnings'].append(f'File validation failed: {file_path}')
                validation_result['details']['step03_files_found'] = len(step03_files)
                validation_result['details']['step03_files'] = [str(f) for f in step03_files]
        except Exception as e:
            validation_result['validation_passed'] = False
            validation_result['errors'].append(f'Prerequisites validation failed: {str(e)}')
        return validation_result

    def validate_outputs(self, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate Step 4 output files and content using BaseValidator methods."""
        validation_result = {'validation_passed': True, 'warnings': [], 'errors': [], 'details': {}}
        try:
            output_dir = Path('generated/market_analysis') / exchange.lower() / symbol.lower() / 'regime_splits'
            expected_files = [f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet']
            missing_files = []
            existing_files = []
            for filename in expected_files:
                file_path = output_dir / filename
                try:
                    file_valid, file_metrics = self.validate_file_exists(str(file_path), f'expected file: {filename}')
                except Exception:
                    file_valid = file_path.exists()
                if file_valid:
                    existing_files.append(str(file_path))
                else:
                    missing_files.append(filename)
            if missing_files:
                validation_result['validation_passed'] = False
                validation_result['errors'].extend([f'Missing regime data splitting file: {f}' for f in missing_files])
            else:
                validation_result['details']['files_found'] = len(existing_files)
                validation_result['details']['files'] = existing_files
            if existing_files:
                for file_path in existing_files:
                    if file_path.endswith('.parquet'):
                        try:
                            # For processed data validation, use safe parquet reader
                            df = safe_read_parquet(file_path)
                            try:
                                df_valid, df_metrics = self.validate_dataframe_quality(df, min_rows = 100, check_data_types = True)
                            except Exception:
                                df_valid = len(df) >= 100
                            validation_result['details'][f'{Path(file_path).stem}_rows'] = len(df)
                            validation_result['details'][f'{Path(file_path).stem}_columns'] = list(df.columns)
                            validation_result['details'][f'{Path(file_path).stem}_valid'] = df_valid
                        except Exception as e:
                            validation_result['warnings'].append(f'Could not read parquet file {file_path}: {e}')
        except Exception as e:
            validation_result['validation_passed'] = False
            validation_result['errors'].append(f'Output validation failed: {str(e)}')
        return validation_result

async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run validation for Step 4: Regime Data Splitting."

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    logger.info('🔍 Validating Step 4: Regime Data Splitting')
    try:
        symbol = training_input.get('symbol', 'ETHUSDT')
        exchange = training_input.get('exchange', 'BINANCE')
        timeframe = training_input.get('timeframe', '1m')
        data_dir = training_input.get('data_dir', 'historical_data')
        config = training_input.get('config', {})
        validator = Step4RegimeDataSplittingValidator(config)
        prereq_result = validator.validate_prerequisites(symbol, exchange, timeframe)
        step_result = await validator.validate_step4_regime_data_splitting(symbol, exchange, data_dir, training_input)
        output_result = validator.validate_outputs(symbol, exchange, timeframe)
        validation_passed = prereq_result['validation_passed'] and step_result and output_result['validation_passed']
        return {'step_name': 'step04_regime_data_splitting', 'validation_passed': validation_passed, 'prerequisites': prereq_result, 'step_execution': step_result, 'outputs': output_result, 'warnings': prereq_result['warnings'] + output_result['warnings'], 'errors': prereq_result['errors'] + output_result['errors']}
    except Exception as e:
        error_context = {'step': 'step04_regime_data_splitting', 'symbol': training_input.get('symbol', 'UNKNOWN'), 'exchange': training_input.get('exchange', 'UNKNOWN'), 'error_type': type(e).__name__, 'error_message': str(e), 'timestamp': pd.Timestamp.now().isoformat()}
        logger.exception(f'❌ Step 4 validation failed: {error_context}')
        return {'step_name': 'step04_regime_data_splitting', 'validation_passed': False, 'error': str(e), 'error_context': error_context}
if __name__ == '__main__':
    import asyncio
    test_input = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m', 'data_dir': 'data_cache', 'config': {}}
    test_state = {}
    result = asyncio.run(run_validator(test_input, test_state))
    tprint(json.dumps(result, indent = 2))