"""
Basic Backtesting Pre-Step

This module provides the pre-processing step for basic backtesting operations.
It handles data loading, validation, and preparation before the main backtesting execution.

Key Features:
- Data loading and validation
- Configuration preparation
- Artifact management
- Data preview and logging
"""

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning, tprint_data_preview
from src.config.pipeline_modes import get_mode_config, get_mode_lookback_days

# Import our custom types
from ..step_types import (
    StepConfig, ExecutionResult, ValidationResult, MetricsDict, DataLoadResult,
    DataFrameType, SeriesType, SignalType, ModelType, DirectionType, ExecutionMode,
    ValidationError, DataLoadError, ConfigurationError, ArtifactError,
    validate_config, create_error_result, create_success_result,
    is_dataframe, is_series, is_valid_config
)

logger = logging.getLogger(__name__)

class BasicBacktestingPreStep(BaseStep):
    """
    Pre-processing step for basic backtesting operations.
    
    This step handles:
    - Loading historical data
    - Validating data quality
    - Preparing configuration
    - Setting up artifacts
    """

    def __init__(self, step_name: str = "basic_backtesting_pre", config: Optional[StepConfig] = None):
        """Initialize the basic backtesting pre-step."""
        super().__init__(step_name, config)
        self.logger = logging.getLogger(f"ares.step.{step_name}")

    async def execute(self, config: StepConfig) -> ExecutionResult:
        """
        Execute the basic backtesting pre-processing step with comprehensive error handling.

        Args:
            config: Configuration containing symbol, exchange, timeframes, etc.

        Returns:
            ExecutionResult with artifacts and metrics

        Raises:
            ValidationError: If data validation fails
            DataLoadError: If data loading fails
            ConfigurationError: If configuration is invalid
            ArtifactError: If artifact operations fail
        """
        try:
            # Validate configuration
            if not is_valid_config(config):
                raise ConfigurationError("Invalid configuration provided")
            
            validated_config = validate_config(config)
            
            tprint_info('🔧 Starting Basic Backtesting Pre-Processing')
            
            # Extract configuration with validation
            symbol = validated_config['symbol']
            exchange = validated_config['exchange']
            timeframe = validated_config['timeframe']
            direction = validated_config['direction']
            execution_mode = validated_config['execution_mode']
            
            # Get mode configuration for lookback periods and other parameters
            mode_config = get_mode_config(execution_mode)
            
            # Preview configuration data
            tprint_data_preview(validated_config, "basic_backtesting_pre_config", max_rows=10, level="DEBUG")
            
            tprint_info(f"Pre-processing backtesting data for {symbol} from {exchange}")
            tprint_info(f"Timeframe: {timeframe}, Direction: {direction}, Mode: {execution_mode}")
            
            # Initialize artifacts list
            artifacts: List[str] = []
            metrics: MetricsDict = {}
            
            # Set up artifact manager context
            try:
                self.artifact_manager.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    direction=direction,
                    model='BasicBacktesting'
                )
                tprint_success("✅ Artifact manager context set")
            except Exception as e:
                raise ArtifactError(f"Failed to set artifact manager context: {e}") from e
            
            # Load historical data
            historical_data = await self._load_historical_data(symbol, exchange, timeframe, validated_config)
            if historical_data is None:
                raise DataLoadError("Failed to load historical data")
            
            # Preview loaded historical data
            tprint_data_preview(historical_data, "loaded_historical_data", max_rows=5, level="INFO")
            
            # Validate data quality
            validation_result = await self._validate_data_quality(historical_data, validated_config)
            tprint_data_preview(validation_result, "data_validation_result", max_rows=5, level="INFO")
            
            if not validation_result.get('is_valid', False):
                raise ValidationError(f"Data validation failed: {validation_result.get('errors', [])}")
            
            # Prepare backtesting configuration
            backtest_config = await self._prepare_backtest_config(validated_config, historical_data)
            tprint_data_preview(backtest_config, "prepared_backtest_config", max_rows=10, level="DEBUG")
            
            # Load model artifacts if available
            model_artifacts = await self._load_model_artifacts(symbol, exchange, direction, validated_config)
            if model_artifacts:
                tprint_data_preview(model_artifacts, "loaded_model_artifacts", max_rows=5, level="DEBUG")
            
            # Prepare data for backtesting
            prepared_data = await self._prepare_data_for_backtesting(historical_data, model_artifacts, validated_config)
            tprint_data_preview(prepared_data, "prepared_backtest_data", max_rows=5, level="INFO")
            
            # Save prepared data as artifact
            try:
                artifact_path = self._save_artifact(
                    prepared_data,
                    'basic_backtesting_prepared_data',
                    'data'
                )
                artifacts.append(artifact_path)
                tprint_success(f"💾 Saved prepared data artifact: {artifact_path}")
            except Exception as e:
                raise ArtifactError(f"Failed to save prepared data artifact: {e}") from e
            
            # Save backtest configuration as artifact
            try:
                config_artifact_path = self._save_artifact(
                    backtest_config,
                    'basic_backtesting_config',
                    'metadata'
                )
                artifacts.append(config_artifact_path)
                tprint_success(f"💾 Saved config artifact: {config_artifact_path}")
            except Exception as e:
                raise ArtifactError(f"Failed to save config artifact: {e}") from e
            
            # Calculate metrics
            metrics = {
                'data_points': len(historical_data),
                'data_quality_score': validation_result.get('quality_score', 0.0),
                'preparation_time': 0.0,  # Will be set by the base step
                'artifacts_created': len(artifacts),
                'validation_errors': len(validation_result.get('errors', [])),
                'validation_warnings': len(validation_result.get('warnings', []))
            }
            
            tprint_success(f"✅ Basic backtesting pre-processing completed successfully")
            tprint_info(f"📊 Data points: {metrics['data_points']}")
            tprint_info(f"📊 Data quality score: {metrics['data_quality_score']:.2f}")
            tprint_info(f"📁 Artifacts created: {metrics['artifacts_created']}")
            
            return create_success_result(
                artifacts=artifacts,
                metrics=metrics,
                data=prepared_data,
                backtest_config=backtest_config
            )
            
        except ValidationError:
            raise
        except DataLoadError:
            raise
        except ConfigurationError:
            raise
        except ArtifactError:
            raise
        except Exception as e:
            tprint_error(f"❌ Unexpected error in basic backtesting pre-processing: {e}")
            return create_error_result(e, "basic_backtesting_pre_execute")

    async def _load_historical_data(self, symbol: str, exchange: str, timeframe: str, config: StepConfig) -> Optional[DataFrameType]:
        """Load historical data for backtesting."""
        try:
            self.logger.info(f"📂 Loading historical data: {symbol} {exchange} {timeframe}")
            
            # Try to load from klines manager first
            if self._is_klines_available():
                historical_data = self._load_klines_with_context(timeframe)
                if historical_data is not None:
                    self.logger.info(f"✅ Loaded historical data from klines manager: {len(historical_data)} records")
                    return historical_data
            
            # Fallback to artifact manager
            historical_data = self._load_dataframe('historical_data')
            if historical_data is not None:
                self.logger.info(f"✅ Loaded historical data from artifacts: {len(historical_data)} records")
                return historical_data
            
            # Try to load from different artifact names
            possible_names = [
                'klines_data',
                'market_data',
                'price_data',
                f'{symbol}_{timeframe}_data',
                f'{exchange}_{symbol}_data'
            ]
            
            for name in possible_names:
                historical_data = self._load_dataframe(name)
                if historical_data is not None:
                    self.logger.info(f"✅ Loaded historical data as '{name}': {len(historical_data)} records")
                    return historical_data
            
            self.logger.warning("⚠️ No historical data found")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load historical data: {e}")
            return None

    async def _validate_data_quality(self, data: DataFrameType, config: StepConfig) -> ValidationResult:
        """Validate data quality for backtesting."""
        try:
            self.logger.info("🔍 Validating data quality")
            
            validation_result = {
                'is_valid': True,
                'quality_score': 0.0,
                'errors': [],
                'warnings': []
            }
            
            if data is None or len(data) == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Data is None or empty")
                return validation_result
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                return validation_result
            
            # Check for NaN values
            nan_counts = data.isnull().sum()
            if nan_counts.any():
                validation_result['warnings'].append(f"Found NaN values: {nan_counts.to_dict()}")
            
            # Check for duplicate timestamps
            if 'timestamp' in data.columns:
                duplicate_timestamps = data['timestamp'].duplicated().sum()
                if duplicate_timestamps > 0:
                    validation_result['warnings'].append(f"Found {duplicate_timestamps} duplicate timestamps")
            
            # Check data consistency
            invalid_ohlc = (data['high'] < data['low']).sum()
            if invalid_ohlc > 0:
                validation_result['warnings'].append(f"Found {invalid_ohlc} invalid OHLC combinations")
            
            # Calculate quality score
            quality_score = 1.0
            quality_score -= len(validation_result['errors']) * 0.3
            quality_score -= len(validation_result['warnings']) * 0.1
            quality_score = max(0.0, quality_score)
            validation_result['quality_score'] = quality_score
            
            self.logger.info(f"✅ Data validation completed: quality score = {quality_score:.2f}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            return {
                'is_valid': False,
                'quality_score': 0.0,
                'errors': [str(e)],
                'warnings': []
            }

    async def _prepare_backtest_config(self, config: StepConfig, historical_data: DataFrameType) -> Dict[str, Any]:
        """Prepare backtesting configuration."""
        try:
            self.logger.info("⚙️ Preparing backtesting configuration")
            
            backtest_config = {
                'symbol': config.get('symbol', 'ETHUSDT'),
                'exchange': config.get('exchange', 'binance'),
                'timeframe': config.get('timeframe', '15m'),
                'direction': config.get('direction', 'longs'),
                'execution_mode': config.get('execution_mode', 'light'),
                'initial_capital': config.get('initial_capital', 100000.0),
                'commission': config.get('commission', 0.001),
                'slippage': config.get('slippage', 0.0005),
                'data_start': historical_data.index[0] if len(historical_data) > 0 else None,
                'data_end': historical_data.index[-1] if len(historical_data) > 0 else None,
                'data_points': len(historical_data),
                'prepared_at': datetime.now().isoformat()
            }
            
            # Add strategy-specific configuration
            if 'strategy_config' in config:
                backtest_config['strategy_config'] = config['strategy_config']
            
            # Add risk management configuration
            if 'risk_config' in config:
                backtest_config['risk_config'] = config['risk_config']
            
            self.logger.info(f"✅ Backtesting configuration prepared")
            return backtest_config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare backtesting configuration: {e}")
            return {}

    async def _load_model_artifacts(self, symbol: str, exchange: str, direction: str, config: StepConfig) -> Optional[Dict[str, Any]]:
        """Load model artifacts for backtesting."""
        try:
            self.logger.info("📂 Loading model artifacts")
            
            model_artifacts = {}
            
            # Try to load different model types
            model_types = ['analyst', 'tactician', 'ensemble']
            
            for model_type in model_types:
                model_data = self._load_dataframe(f'{model_type}_model')
                if model_data is not None:
                    model_artifacts[model_type] = model_data
                    self.logger.info(f"✅ Loaded {model_type} model artifacts")
            
            # Try to load predictions
            predictions = self._load_dataframe('model_predictions')
            if predictions is not None:
                model_artifacts['predictions'] = predictions
                self.logger.info("✅ Loaded model predictions")
            
            # Try to load features
            features = self._load_dataframe('features')
            if features is not None:
                model_artifacts['features'] = features
                self.logger.info("✅ Loaded features")
            
            if not model_artifacts:
                self.logger.warning("⚠️ No model artifacts found")
                return None
            
            return model_artifacts
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model artifacts: {e}")
            return None

    async def _prepare_data_for_backtesting(self, historical_data: DataFrameType, model_artifacts: Optional[Dict[str, Any]], config: StepConfig) -> Dict[str, Any]:
        """Prepare data for backtesting."""
        try:
            self.logger.info("🔧 Preparing data for backtesting")
            
            prepared_data = {
                'historical_data': historical_data,
                'prepared_at': datetime.now().isoformat()
            }
            
            # Add model artifacts if available
            if model_artifacts:
                prepared_data['model_artifacts'] = model_artifacts
            
            # Add metadata
            prepared_data['metadata'] = {
                'symbol': config.get('symbol', 'ETHUSDT'),
                'exchange': config.get('exchange', 'binance'),
                'timeframe': config.get('timeframe', '15m'),
                'direction': config.get('direction', 'longs'),
                'data_points': len(historical_data),
                'preparation_time': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Data prepared for backtesting: {len(historical_data)} records")
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare data for backtesting: {e}")
            return {}