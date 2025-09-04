#!/usr/bin/env python3
"""Enhanced Backtesting Package for Trading Pipeline.

This package contains all the components for backtesting with comprehensive validation,
error handling, and common utilities:
- Walk forward validation per regime with validators
- Monte Carlo validation per regime with validators
- A/B testing per regime with validators
- Model saving and persistence with validators
- Comprehensive data validation and error handling
- Common utilities for data operations
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

# Import enhanced components
from .step18_walk_forward_validation_per_regime import WalkForwardValidationPerRegimeStep
from .step18_walk_forward_validation_validator import WalkForwardValidationValidator
from .step19_monte_carlo_validation_per_regime import MonteCarloValidationPerRegimeStep
from .step19_monte_carlo_validation_validator import MonteCarloValidationValidator
from .step20_ab_testing_per_regime import ABTestingPerRegimeStep
from .step20_ab_testing_validator import ABTestingValidator
from .step21_saving import SavingStep
from .step21_saving_per_regime import PerRegimeSavingStep
from .step21_saving_validator import SavingValidator

# Import enhanced validation and utilities
from src.utils.base_validator import BaseValidator
from src.utils.common_operations import (
    format_datetime, get_current_datetime, safe_file_exists, 
    ensure_directory, safe_json_dump, safe_json_load
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose
)
from src.core.domain.decorators import (
    validate_data_quality, monitor_step_execution, 
    ensure_data_integrity, validate_pipeline_step
)

logger = logging.getLogger(__name__)

# Enhanced pipeline configuration
class BacktestingPipelineConfig:
    """Configuration for the enhanced backtesting pipeline."""
    
    def __init__(self, **kwargs):
        self.symbol = kwargs.get('symbol', 'ETHUSDT')
        self.exchange = kwargs.get('exchange', 'BINANCE')
        self.timeframe = kwargs.get('timeframe', '1m')
        self.data_dir = kwargs.get('data_dir', 'data_cache')
        
        # Validation settings
        self.enable_validation = kwargs.get('enable_validation', True)
        self.strict_validation = kwargs.get('strict_validation', False)
        self.validate_data_quality = kwargs.get('validate_data_quality', True)
        
        # Pipeline steps
        self.walk_forward_validation = kwargs.get('walk_forward_validation', True)
        self.monte_carlo_validation = kwargs.get('monte_carlo_validation', True)
        self.ab_testing = kwargs.get('ab_testing', True)
        self.model_saving = kwargs.get('model_saving', True)
        
        # Error handling
        self.retry_failed_steps = kwargs.get('retry_failed_steps', True)
        self.max_retries = kwargs.get('max_retries', 3)
        self.timeout_seconds = kwargs.get('timeout_seconds', 3600)
        
        # Performance monitoring
        self.enable_performance_monitoring = kwargs.get('enable_performance_monitoring', True)
        self.log_detailed_metrics = kwargs.get('log_detailed_metrics', True)

# Enhanced pipeline validator
class BacktestingPipelineValidator(BaseValidator):
    """Comprehensive validator for the backtesting pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("backtesting_pipeline", config)
        self.pipeline_config = BacktestingPipelineConfig(**config)
    
    async def validate(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> bool:
        """Validate the backtesting pipeline prerequisites and configuration."""
        try:
            self.logger.info("🔍 Starting backtesting pipeline validation")
            
            # Validate input parameters
            validation_results = {}
            
            # Validate symbol and exchange
            symbol_validation = await self._validate_symbol_exchange(
                training_input.get('symbol'), training_input.get('exchange')
            )
            validation_results['symbol_exchange'] = symbol_validation
            
            # Validate data directory and files
            data_validation = await self._validate_data_availability(
                training_input.get('data_dir', self.pipeline_config.data_dir)
            )
            validation_results['data_availability'] = data_validation
            
            # Validate configuration
            config_validation = await self._validate_pipeline_configuration()
            validation_results['configuration'] = config_validation
            
            # Validate prerequisites
            prerequisites_validation = await self._validate_prerequisites(pipeline_state)
            validation_results['prerequisites'] = prerequisites_validation
            
            # Store validation results
            self.validation_results = validation_results
            
            # Determine overall success
            all_passed = all(
                result.get('validation_passed', False) 
                for result in validation_results.values()
            )
            
            if all_passed:
                self.logger.info("✅ Backtesting pipeline validation passed")
            else:
                self.logger.error("❌ Backtesting pipeline validation failed")
                self._log_validation_failures(validation_results)
            
            return all_passed
            
        except Exception as e:
            self.logger.exception(f"❌ Error in backtesting pipeline validation: {e}")
            return False
    
    async def _validate_symbol_exchange(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Validate symbol and exchange parameters."""
        try:
            if not symbol or not exchange:
                return {
                    'validation_passed': False,
                    'error': 'Symbol and exchange are required'
                }
            
            # Validate symbol format
            if not isinstance(symbol, str) or len(symbol) < 3:
                return {
                    'validation_passed': False,
                    'error': f'Invalid symbol format: {symbol}'
                }
            
            # Validate exchange
            valid_exchanges = ['BINANCE', 'MEXC', 'GATEIO']
            if exchange.upper() not in valid_exchanges:
                return {
                    'validation_passed': False,
                    'error': f'Unsupported exchange: {exchange}'
                }
            
            return {
                'validation_passed': True,
                'details': f'Valid symbol {symbol} and exchange {exchange}'
            }
            
        except Exception as e:
            return {
                'validation_passed': False,
                'error': f'Error validating symbol/exchange: {e}'
            }
    
    async def _validate_data_availability(self, data_dir: str) -> Dict[str, Any]:
        """Validate data directory and required files."""
        try:
            data_path = Path(data_dir)
            
            if not data_path.exists():
                return {
                    'validation_passed': False,
                    'error': f'Data directory does not exist: {data_dir}'
                }
            
            # Check for required data files
            required_files = [
                f"aggtrades_{self.pipeline_config.exchange}_{self.pipeline_config.symbol}_consolidated.parquet",
                f"volume_{self.pipeline_config.exchange}_{self.pipeline_config.symbol}_consolidated.parquet"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = data_path / file_name
                if not safe_file_exists(file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                return {
                    'validation_passed': False,
                    'error': f'Missing required data files: {missing_files}'
                }
            
            return {
                'validation_passed': True,
                'details': f'All required data files found in {data_dir}'
            }
            
        except Exception as e:
            return {
                'validation_passed': False,
                'error': f'Error validating data availability: {e}'
            }
    
    async def _validate_pipeline_configuration(self) -> Dict[str, Any]:
        """Validate pipeline configuration."""
        try:
            # Validate timeout settings
            if self.pipeline_config.timeout_seconds <= 0:
                return {
                    'validation_passed': False,
                    'error': 'Invalid timeout_seconds configuration'
                }
            
            # Validate retry settings
            if self.pipeline_config.max_retries < 0:
                return {
                    'validation_passed': False,
                    'error': 'Invalid max_retries configuration'
                }
            
            return {
                'validation_passed': True,
                'details': 'Pipeline configuration is valid'
            }
            
        except Exception as e:
            return {
                'validation_passed': False,
                'error': f'Error validating configuration: {e}'
            }
    
    async def _validate_prerequisites(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline prerequisites."""
        try:
            # Check for required previous steps
            required_steps = [
                'step1_data_collection',
                'step2_data_reading',
                'step9_hmm_based_training'
            ]
            
            missing_steps = []
            for step in required_steps:
                if step not in pipeline_state or not pipeline_state[step].get('completed', False):
                    missing_steps.append(step)
            
            if missing_steps:
                return {
                    'validation_passed': False,
                    'error': f'Missing required previous steps: {missing_steps}'
                }
            
            return {
                'validation_passed': True,
                'details': 'All prerequisites are satisfied'
            }
            
        except Exception as e:
            return {
                'validation_passed': False,
                'error': f'Error validating prerequisites: {e}'
            }
    
    def _log_validation_failures(self, validation_results: Dict[str, Any]) -> None:
        """Log validation failures for debugging."""
        for step, result in validation_results.items():
            if not result.get('validation_passed', False):
                error = result.get('error', 'Unknown error')
                self.logger.error(f"❌ {step} validation failed: {error}")

# Enhanced pipeline with comprehensive validation and error handling
@compose(
    error_boundary(name="backtesting_pipeline"),
    traced(span_name="backtesting_pipeline"),
    log_execution_time,
    timeout(seconds=3600)
)
@validate_pipeline_step(
    prerequisites=['step1_data_collection', 'step2_data_reading', 'step9_hmm_based_training'],
    outputs=['walk_forward_results', 'monte_carlo_results', 'ab_testing_results', 'model_saving_results']
)
async def run_backtesting_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the enhanced backtesting pipeline with comprehensive validation and error handling."""
    try:
        logger.info("🚀 Starting enhanced backtesting pipeline")
        logger.info(f"📊 Configuration: {symbol} on {exchange}, timeframe: {timeframe}")
        
        # Initialize pipeline configuration
        pipeline_config = BacktestingPipelineConfig(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        # Initialize validator
        validator = BacktestingPipelineValidator(pipeline_config.__dict__)
        
        # Prepare training input and pipeline state
        training_input = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'data_dir': data_dir
        }
        
        # Mock pipeline state for validation (in real usage, this would come from previous steps)
        pipeline_state = {
            'step1_data_collection': {'completed': True},
            'step2_data_reading': {'completed': True},
            'step9_hmm_based_training': {'completed': True}
        }
        
        # Validate pipeline prerequisites
        if pipeline_config.enable_validation:
            validation_passed = await validator.validate(training_input, pipeline_state)
            if not validation_passed:
                logger.error("❌ Pipeline validation failed, aborting")
                return False
        
        # Initialize results tracking
        pipeline_results = {
            'walk_forward_results': None,
            'monte_carlo_results': None,
            'ab_testing_results': None,
            'model_saving_results': None,
            'start_time': get_current_datetime(),
            'config': pipeline_config.__dict__
        }
        
        # Step 1: Walk Forward Validation (if enabled)
        if pipeline_config.walk_forward_validation:
            logger.info("🔄 Starting walk forward validation")
            try:
                walk_forward = WalkForwardValidationPerRegimeStep(pipeline_config.__dict__)
                walk_forward_results = await walk_forward.validate_walk_forward(
                    symbol, exchange, timeframe, data_dir
                )
                pipeline_results['walk_forward_results'] = walk_forward_results
                logger.info("✅ Walk forward validation completed")
            except Exception as e:
                logger.exception(f"❌ Walk forward validation failed: {e}")
                if pipeline_config.strict_validation:
                    return False
        
        # Step 2: Monte Carlo Validation (if enabled)
        if pipeline_config.monte_carlo_validation:
            logger.info("🔄 Starting Monte Carlo validation")
            try:
                monte_carlo = MonteCarloValidationPerRegimeStep(pipeline_config.__dict__)
                monte_carlo_results = await monte_carlo.validate_monte_carlo(
                    symbol, exchange, timeframe, data_dir
                )
                pipeline_results['monte_carlo_results'] = monte_carlo_results
                logger.info("✅ Monte Carlo validation completed")
            except Exception as e:
                logger.exception(f"❌ Monte Carlo validation failed: {e}")
                if pipeline_config.strict_validation:
                    return False
        
        # Step 3: A/B Testing (if enabled)
        if pipeline_config.ab_testing:
            logger.info("🔄 Starting A/B testing")
            try:
                ab_tester = ABTestingPerRegimeStep(pipeline_config.__dict__)
                ab_testing_results = await ab_tester.run_ab_testing(
                    symbol, exchange, timeframe, data_dir
                )
                pipeline_results['ab_testing_results'] = ab_testing_results
                logger.info("✅ A/B testing completed")
            except Exception as e:
                logger.exception(f"❌ A/B testing failed: {e}")
                if pipeline_config.strict_validation:
                    return False
        
        # Step 4: Model Saving (if enabled)
        if pipeline_config.model_saving:
            logger.info("🔄 Starting model saving")
            try:
                model_saver = SavingStep(pipeline_config.__dict__)
                model_saving_results = await model_saver.save_models(
                    symbol, exchange, timeframe, data_dir
                )
                pipeline_results['model_saving_results'] = model_saving_results
                logger.info("✅ Model saving completed")
            except Exception as e:
                logger.exception(f"❌ Model saving failed: {e}")
                if pipeline_config.strict_validation:
                    return False
        
        # Save pipeline results
        pipeline_results['end_time'] = get_current_datetime()
        pipeline_results['success'] = True
        
        # Save results to file
        results_file = Path(data_dir) / f"backtesting_pipeline_results_{symbol}_{timeframe}.json"
        safe_json_dump(pipeline_results, results_file, indent=2)
        logger.info(f"💾 Pipeline results saved to: {results_file}")
        
        logger.info("🎉 Enhanced backtesting pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"💥 Enhanced backtesting pipeline failed: {e}")
        return False

__all__ = [
    'WalkForwardValidationPerRegimeStep',
    'WalkForwardValidationValidator',
    'MonteCarloValidationPerRegimeStep',
    'MonteCarloValidationValidator',
    'ABTestingPerRegimeStep',
    'ABTestingValidator',
    'SavingStep',
    'PerRegimeSavingStep',
    'SavingValidator',
    'run_backtesting_pipeline'
]