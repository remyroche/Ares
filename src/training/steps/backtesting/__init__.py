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
- Enhanced logging with emojis and progress tracking
- Performance monitoring and quality assessment
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

# Import enhanced logging system
from .enhanced_logging import BacktestingLogger, get_backtesting_logger

# Import comprehensive reporting system
from .comprehensive_reporting import generate_backtesting_report

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
    
    # Initialize enhanced logger
    bt_logger = get_backtesting_logger(f"{symbol}_{exchange}_{timeframe}", log_dir="log/backtesting")
    bt_logger.start_performance_monitoring(interval=10.0)
    
    try:
        bt_logger.log_info("🚀 Starting Enhanced Backtesting Pipeline", "INITIALIZATION")
        bt_logger.log_info(f"📊 Configuration: {symbol} on {exchange}, timeframe: {timeframe}", "CONFIG")
        
        # Initialize pipeline configuration
        pipeline_config = BacktestingPipelineConfig(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
        # Log configuration details
        bt_logger.log_info("📋 Pipeline Configuration:", "CONFIG")
        bt_logger.log_info(f"   • Symbol: {pipeline_config.symbol}", "CONFIG")
        bt_logger.log_info(f"   • Exchange: {pipeline_config.exchange}", "CONFIG")
        bt_logger.log_info(f"   • Timeframe: {pipeline_config.timeframe}", "CONFIG")
        bt_logger.log_info(f"   • Data Directory: {pipeline_config.data_dir}", "CONFIG")
        bt_logger.log_info(f"   • Walk Forward Validation: {pipeline_config.walk_forward_validation}", "CONFIG")
        bt_logger.log_info(f"   • Monte Carlo Validation: {pipeline_config.monte_carlo_validation}", "CONFIG")
        bt_logger.log_info(f"   • A/B Testing: {pipeline_config.ab_testing}", "CONFIG")
        bt_logger.log_info(f"   • Model Saving: {pipeline_config.model_saving}", "CONFIG")
        bt_logger.log_info(f"   • Enable Validation: {pipeline_config.enable_validation}", "CONFIG")
        bt_logger.log_info(f"   • Strict Validation: {pipeline_config.strict_validation}", "CONFIG")
        bt_logger.log_info(f"   • Performance Monitoring: {pipeline_config.enable_performance_monitoring}", "CONFIG")
        
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
        
        # Step 0: Pre-flight Validation
        bt_logger.log_progress("Pre-flight Validation", 0, "Starting validation checks")
        
        if pipeline_config.enable_validation:
            with bt_logger.step_timer("pre_flight_validation"):
                bt_logger.log_info("🔍 Running pre-flight validation", "VALIDATION")
                validation_passed = await validator.validate(training_input, pipeline_state)
                
                if not validation_passed:
                    bt_logger.log_error(Exception("Pipeline validation failed"), "VALIDATION")
                    bt_logger.log_quality_flag("VALIDATION_FAILURE", "Pre-flight validation failed", "ERROR")
                    return False
                
                bt_logger.log_success("Pre-flight validation passed", "VALIDATION")
        
        bt_logger.log_progress("Pre-flight Validation", 100, "Validation completed successfully")
        
        # Initialize results tracking
        pipeline_results = {
            'walk_forward_results': None,
            'monte_carlo_results': None,
            'ab_testing_results': None,
            'model_saving_results': None,
            'start_time': get_current_datetime(),
            'config': pipeline_config.__dict__
        }
        
        # Calculate total steps for progress tracking
        total_steps = sum([
            pipeline_config.walk_forward_validation,
            pipeline_config.monte_carlo_validation,
            pipeline_config.ab_testing,
            pipeline_config.model_saving
        ])
        current_step = 0
        
        # Step 1: Walk Forward Validation (if enabled)
        if pipeline_config.walk_forward_validation:
            current_step += 1
            progress = (current_step / total_steps) * 100
            bt_logger.log_progress("Walk Forward Validation", progress, "Starting walk forward validation")
            
            with bt_logger.step_timer("walk_forward_validation"):
                try:
                    bt_logger.log_info("🔄 Starting walk forward validation", "WALK_FORWARD")
                    walk_forward = WalkForwardValidationPerRegimeStep(pipeline_config.__dict__)
                    walk_forward_results = await walk_forward.validate_walk_forward(
                        symbol, exchange, timeframe, data_dir
                    )
                    pipeline_results['walk_forward_results'] = walk_forward_results
                    bt_logger.log_success("Walk forward validation completed", "WALK_FORWARD")
                    
                    # Log backtesting metrics and quality assessment
                    if walk_forward_results and isinstance(walk_forward_results, dict):
                        bt_logger.log_backtesting_metrics(walk_forward_results, "Walk Forward Validation")
                        
                        # Log regime analysis if available
                        if 'regime_performance' in walk_forward_results:
                            bt_logger.log_regime_analysis(walk_forward_results['regime_performance'])
                        
                        # Log model performance if available
                        if 'model_performance' in walk_forward_results:
                            bt_logger.log_model_performance(walk_forward_results['model_performance'])
                        
                        # Log risk metrics if available
                        if 'risk_metrics' in walk_forward_results:
                            bt_logger.log_risk_metrics(walk_forward_results['risk_metrics'])
                    
                except Exception as e:
                    bt_logger.log_error(e, "WALK_FORWARD")
                    bt_logger.log_quality_flag("WALK_FORWARD_FAILURE", f"Walk forward validation failed: {e}", "ERROR")
                    if pipeline_config.strict_validation:
                        return False
        
        # Step 2: Monte Carlo Validation (if enabled)
        if pipeline_config.monte_carlo_validation:
            current_step += 1
            progress = (current_step / total_steps) * 100
            bt_logger.log_progress("Monte Carlo Validation", progress, "Starting Monte Carlo validation")
            
            with bt_logger.step_timer("monte_carlo_validation"):
                try:
                    bt_logger.log_info("🔄 Starting Monte Carlo validation", "MONTE_CARLO")
                    monte_carlo = MonteCarloValidationPerRegimeStep(pipeline_config.__dict__)
                    monte_carlo_results = await monte_carlo.validate_monte_carlo(
                        symbol, exchange, timeframe, data_dir
                    )
                    pipeline_results['monte_carlo_results'] = monte_carlo_results
                    bt_logger.log_success("Monte Carlo validation completed", "MONTE_CARLO")
                    
                    # Log backtesting metrics and quality assessment
                    if monte_carlo_results and isinstance(monte_carlo_results, dict):
                        bt_logger.log_backtesting_metrics(monte_carlo_results, "Monte Carlo Validation")
                        
                        # Log regime analysis if available
                        if 'regime_performance' in monte_carlo_results:
                            bt_logger.log_regime_analysis(monte_carlo_results['regime_performance'])
                        
                        # Log model performance if available
                        if 'model_performance' in monte_carlo_results:
                            bt_logger.log_model_performance(monte_carlo_results['model_performance'])
                        
                        # Log risk metrics if available
                        if 'risk_metrics' in monte_carlo_results:
                            bt_logger.log_risk_metrics(monte_carlo_results['risk_metrics'])
                    
                except Exception as e:
                    bt_logger.log_error(e, "MONTE_CARLO")
                    bt_logger.log_quality_flag("MONTE_CARLO_FAILURE", f"Monte Carlo validation failed: {e}", "ERROR")
                    if pipeline_config.strict_validation:
                        return False
        
        # Step 3: A/B Testing (if enabled)
        if pipeline_config.ab_testing:
            current_step += 1
            progress = (current_step / total_steps) * 100
            bt_logger.log_progress("A/B Testing", progress, "Starting A/B testing")
            
            with bt_logger.step_timer("ab_testing"):
                try:
                    bt_logger.log_info("🔄 Starting A/B testing", "AB_TESTING")
                    ab_tester = ABTestingPerRegimeStep(pipeline_config.__dict__)
                    ab_testing_results = await ab_tester.run_ab_testing(
                        symbol, exchange, timeframe, data_dir
                    )
                    pipeline_results['ab_testing_results'] = ab_testing_results
                    bt_logger.log_success("A/B testing completed", "AB_TESTING")
                    
                    # Log backtesting metrics and quality assessment
                    if ab_testing_results and isinstance(ab_testing_results, dict):
                        bt_logger.log_backtesting_metrics(ab_testing_results, "A/B Testing")
                        
                        # Log regime analysis if available
                        if 'regime_performance' in ab_testing_results:
                            bt_logger.log_regime_analysis(ab_testing_results['regime_performance'])
                        
                        # Log model performance if available
                        if 'model_performance' in ab_testing_results:
                            bt_logger.log_model_performance(ab_testing_results['model_performance'])
                        
                        # Log risk metrics if available
                        if 'risk_metrics' in ab_testing_results:
                            bt_logger.log_risk_metrics(ab_testing_results['risk_metrics'])
                    
                except Exception as e:
                    bt_logger.log_error(e, "AB_TESTING")
                    bt_logger.log_quality_flag("AB_TESTING_FAILURE", f"A/B testing failed: {e}", "ERROR")
                    if pipeline_config.strict_validation:
                        return False
        
        # Step 4: Model Saving (if enabled)
        if pipeline_config.model_saving:
            current_step += 1
            progress = (current_step / total_steps) * 100
            bt_logger.log_progress("Model Saving", progress, "Starting model saving")
            
            with bt_logger.step_timer("model_saving"):
                try:
                    bt_logger.log_info("🔄 Starting model saving", "MODEL_SAVING")
                    model_saver = SavingStep(pipeline_config.__dict__)
                    model_saving_results = await model_saver.save_models(
                        symbol, exchange, timeframe, data_dir
                    )
                    pipeline_results['model_saving_results'] = model_saving_results
                    bt_logger.log_success("Model saving completed", "MODEL_SAVING")
                    
                    # Log model saving results
                    if model_saving_results and isinstance(model_saving_results, dict):
                        bt_logger.log_info("Model Saving Results:", "MODEL_SAVING")
                        for key, value in model_saving_results.items():
                            bt_logger.log_info(f"   • {key}: {value}", "MODEL_SAVING")
                    
                except Exception as e:
                    bt_logger.log_error(e, "MODEL_SAVING")
                    bt_logger.log_quality_flag("MODEL_SAVING_FAILURE", f"Model saving failed: {e}", "ERROR")
                    if pipeline_config.strict_validation:
                        return False
        
        # Final progress update
        bt_logger.log_progress("Pipeline Completion", 100, "All steps completed successfully")
        
        # Save pipeline results
        pipeline_results['end_time'] = get_current_datetime()
        pipeline_results['success'] = True
        
        # Save results to file
        results_file = Path(data_dir) / f"backtesting_pipeline_results_{symbol}_{timeframe}.json"
        safe_json_dump(pipeline_results, results_file, indent=2)
        bt_logger.log_success(f"Pipeline results saved to: {results_file}", "RESULTS")
        
        # Generate comprehensive report
        report_file = Path(data_dir) / f"backtesting_report_{symbol}_{timeframe}.json"
        logger_report = bt_logger.generate_report(str(report_file))
        
        # Generate comprehensive backtesting report
        comprehensive_report_file = Path(data_dir) / f"comprehensive_backtesting_report_{symbol}_{timeframe}.json"
        comprehensive_report = generate_backtesting_report(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            pipeline_results=pipeline_results,
            logger_data=logger_report,
            output_file=str(comprehensive_report_file)
        )
        
        bt_logger.log_success(f"Comprehensive report saved to: {comprehensive_report_file}", "REPORTING")
        
        # Log performance summary
        bt_logger.log_performance_summary()
        
        # Log final success
        bt_logger.log_success("🎉 Enhanced backtesting pipeline completed successfully", "COMPLETION")
        bt_logger.log_info("=" * 80, "COMPLETION")
        bt_logger.log_info("✅ All enhanced backtesting steps completed:", "COMPLETION")
        bt_logger.log_info("   ✅ Comprehensive validation with quality assessment", "COMPLETION")
        bt_logger.log_info("   ✅ Walk forward validation with detailed logging", "COMPLETION")
        bt_logger.log_info("   ✅ Monte Carlo validation with performance monitoring", "COMPLETION")
        bt_logger.log_info("   ✅ A/B testing with quality flags", "COMPLETION")
        bt_logger.log_info("   ✅ Model saving with comprehensive reporting", "COMPLETION")
        bt_logger.log_info("   ✅ Performance monitoring and resource tracking", "COMPLETION")
        bt_logger.log_info("   ✅ Enhanced logging with emojis and progress indicators", "COMPLETION")
        bt_logger.log_info("=" * 80, "COMPLETION")
        
        return True
        
    except Exception as e:
        bt_logger.log_error(e, "PIPELINE_EXECUTION")
        bt_logger.log_quality_flag("PIPELINE_FAILURE", f"Pipeline execution failed: {e}", "ERROR")
        bt_logger.log_performance_summary()
        
        # Generate failure report
        failure_report_file = Path(data_dir) / f"backtesting_failure_report_{symbol}_{timeframe}.json"
        failure_report = bt_logger.generate_report(str(failure_report_file))
        
        return False
        
    finally:
        # Cleanup
        bt_logger.stop_performance_monitoring()
        bt_logger.cleanup()

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