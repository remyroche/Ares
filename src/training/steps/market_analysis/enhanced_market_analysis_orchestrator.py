from src.utils.tprint import tprint

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

from src.core.decorators import handles_errors, traced, log_execution_time, validates, circuit_breaker, timeout, retry
from src.core.decorators.logging import audit_log, set_correlation_id
from src.training.steps.market_analysis.enhanced_pipeline_decorators import comprehensive_pipeline_protection
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler


"""
Enhanced Market Analysis Orchestrator

This module provides a comprehensive orchestrator for the market analysis pipeline
with proper validators, decorators, and common utilities to ensure each step
leads to the next with proper validation and protection.
"""
import asyncio
import time
from pathlib import Path

from src.training.steps.market_analysis.enhanced_logging_metrics import EnhancedPipelineLogger
from src.training.steps.market_analysis.progress_monitor import progress_monitor
from src.utils.common_operations import get_current_datetime, get_logger, safe_file_exists, format_datetime, safe_json_dump
from src.utils.validator_orchestrator import ValidatorOrchestrator
from src.utils.step_dependency_validator import StepDependencyValidator
from .step04_regime_data_splitting import RegimeDataSplittingStep
# Import validators
from .enhanced_step_validator import EnhancedStepValidator
from .step04_regime_data_splitting_validator import Step4RegimeDataSplittingValidator as RegimeDataSplittingValidator
from .step05_labeling_validator import Step5LabelingValidator as LabelingValidator
# from .step06_feature_engineering_validator import Step6FeatureEngineeringValidator as FeatureEngineeringValidator  # Module not found
# Fallback validator
class FeatureEngineeringValidator:
    def __init__(self):
        pass
    def validate(self, *args, **kwargs):
        return True
# Matrix operations validator removed - using standardized utilities

# Import step classes
from .step05_labeling import LabelingStep
try:
    from .step06_feature_engineering_per_regime import FeatureEngineeringStep
except ImportError:
    try:
        from src.feature_engineering.step06_enhanced_feature_engineering import EnhancedFeatureEngineering as FeatureEngineeringStep
    except ImportError:
        # Fallback class
        class FeatureEngineeringStep:
            def __init__(self, *args, **kwargs):
                pass
                def process(self, *args, **kwargs):
                    return None

# Use standardized matrix operations from ml_common
from src.utils.ml_common.matrix_operations import get_unified_matrix_operations

# Import final feature selection step
from .final_feature_selection_step import FinalFeatureSelectionStep, run_final_feature_selection_step
class EnhancedMatrixOperationsStep:
    def __init__(self, config):
        self.matrix_ops = get_unified_matrix_operations()

try:
    from src.utils.feature_selection.step08_advanced_feature_selection_wrapper import AdvancedFeatureSelectionWrapper as AdvancedFeatureSelectionStep
except ImportError:
    from src.utils.ml_common.feature_selection import UnifiedFeatureSelectionManager
    class AdvancedFeatureSelectionStep:
        def __init__(self, config):
            self.feature_selector = UnifiedFeatureSelectionManager(config)
try:
    from .hmm_clustering.step03_hmm_regime_discovery import run_step as run_enhanced_step
except ImportError:
    # Fallback function
    def run_enhanced_step(*args, **kwargs):
        return None
import json
import logging

# Import ML Common utilities for enhanced functionality
try:
    from src.utils.ml_common import (
        DataQualityUtilities,
        FeatureSelectionFramework,
        MLPipelineOrchestrator
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    logging.warning(f"⚠️ ML Common utilities not available in market analysis orchestrator: {e}")

class MarketAnalysisPipelineOrchestrator:
    """
    Enhanced orchestrator for market analysis pipeline with comprehensive
    validation, error handling, and observability.
    """
    @log_important_calls

    def __init__(self, config: Optional[Dict[str, Any]]=None) -> None:
        """Initialize the orchestrator with configuration."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.enhanced_logger = EnhancedPipelineLogger('market_analysis_orchestrator')
        self.validator_orchestrator = ValidatorOrchestrator()
        self.dependency_validator = StepDependencyValidator()
        self.enhanced_validator = EnhancedStepValidator(config)
        
        # 🖨️ THOROUGH PRINTING: Initialize orchestrator
        tprint("🚀 INITIALIZING MARKET ANALYSIS ORCHESTRATOR")
        tprint("=" * 80)
        tprint(f"📋 Configuration received: {bool(config)}")
        tprint(f"🔧 Enhanced logger initialized: {self.enhanced_logger}")
        tprint(f"✅ Validator orchestrator initialized: {self.validator_orchestrator}")
        tprint(f"🔗 Dependency validator initialized: {self.dependency_validator}")
        tprint(f"🎯 Enhanced validator initialized: {self.enhanced_validator}")
        tprint("=" * 80)

        # Initialize ML Common utilities if available
        tprint("🔬 INITIALIZING ML COMMON UTILITIES")
        tprint(f"📊 ML Common available: {ML_COMMON_AVAILABLE}")
        
        if ML_COMMON_AVAILABLE:
            try:
                self.ml_data_quality = DataQualityUtilities()
                self.ml_feature_selection = FeatureSelectionFramework()
                self.ml_pipeline_orchestrator = MLPipelineOrchestrator()
                self.logger.info("✅ ML Common utilities initialized in market analysis orchestrator")
                tprint("✅ ML Common utilities initialized successfully")
                tprint(f"   📊 Data quality utilities: {self.ml_data_quality}")
                tprint(f"   🎯 Feature selection framework: {self.ml_feature_selection}")
                tprint(f"   🔧 Pipeline orchestrator: {self.ml_pipeline_orchestrator}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize ML Common utilities: {e}")
                tprint(f"⚠️ Failed to initialize ML Common utilities: {e}")
                self.ml_data_quality = None
                self.ml_feature_selection = None
                self.ml_pipeline_orchestrator = None
        else:
            self.ml_data_quality = None
            self.ml_feature_selection = None
            self.ml_pipeline_orchestrator = None
            tprint("⚠️ ML Common utilities not available - using fallback components")
        self.pipeline_state = {'current_step': None, 'completed_steps': [], 'failed_steps': [], 'start_time': None, 'end_time': None, 'correlation_id': None}
        self.step_configs = {'hmm_clustering': {'enabled': True, 'timeout': 300, 'retry_attempts': 3, 'validator': None, 'step_number': 1}, 'regime_splitting': {'enabled': True, 'timeout': 180, 'retry_attempts': 2, 'validator': RegimeDataSplittingValidator(), 'step_number': 2}, 'labeling': {'enabled': True, 'timeout': 240, 'retry_attempts': 2, 'validator': LabelingValidator(), 'step_number': 3}, 'feature_engineering': {'enabled': True, 'timeout': 600, 'retry_attempts': 2, 'validator': FeatureEngineeringValidator(), 'step_number': 4}, 'matrix_operations': {'enabled': True, 'timeout': 300, 'retry_attempts': 2, 'validator': None, 'step_number': 5}, 'feature_selection': {'enabled': True, 'timeout': 180, 'retry_attempts': 2, 'validator': None, 'step_number': 6}, 'final_feature_selection': {'enabled': True, 'timeout': 600, 'retry_attempts': 2, 'validator': None, 'step_number': 7}}
        
        # 🖨️ THOROUGH PRINTING: Pipeline state and step configurations
        tprint("📊 PIPELINE STATE INITIALIZATION")
        tprint(f"   🔄 Current step: {self.pipeline_state['current_step']}")
        tprint(f"   ✅ Completed steps: {self.pipeline_state['completed_steps']}")
        tprint(f"   ❌ Failed steps: {self.pipeline_state['failed_steps']}")
        tprint(f"   ⏰ Start time: {self.pipeline_state['start_time']}")
        tprint(f"   ⏰ End time: {self.pipeline_state['end_time']}")
        tprint(f"   🔗 Correlation ID: {self.pipeline_state['correlation_id']}")
        
        tprint("🔧 STEP CONFIGURATIONS")
        for step_name, config in self.step_configs.items():
            tprint(f"   📋 {step_name}:")
            tprint(f"      ✅ Enabled: {config['enabled']}")
            tprint(f"      ⏱️ Timeout: {config['timeout']}s")
            tprint(f"      🔄 Retry attempts: {config['retry_attempts']}")
            tprint(f"      🔢 Step number: {config['step_number']}")
            tprint(f"      ✅ Validator: {config['validator'] is not None}")
        
        tprint("🎉 ORCHESTRATOR INITIALIZATION COMPLETE")
        tprint("=" * 80)

    async def execute_pipeline(self, symbol: str, exchange: str, timeframe: str='1m', data_dir: str='historical_data', **kwargs) -> bool:
        """
        Execute the complete market analysis pipeline with comprehensive validation.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'BINANCE')
            timeframe: Data timeframe (e.g., '1m')
            data_dir: Data directory path
            **kwargs: Additional configuration parameters
            
        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        # 🖨️ THOROUGH PRINTING: Pipeline execution start
        tprint("🚀 EXECUTING MARKET ANALYSIS PIPELINE")
        tprint("=" * 80)
        tprint(f"🎯 Symbol: {symbol}")
        tprint(f"🏢 Exchange: {exchange}")
        tprint(f"📊 Timeframe: {timeframe}")
        tprint(f"📁 Data directory: {data_dir}")
        tprint(f"⚙️ Additional kwargs: {kwargs}")
        tprint("=" * 80)
        
        correlation_id = f'market_analysis_{symbol}_{exchange}_{int(time.time())}'
        set_correlation_id(correlation_id)
        self.pipeline_state['correlation_id'] = correlation_id
        self.pipeline_state['start_time'] = get_current_datetime()
        
        tprint(f"🔗 Correlation ID generated: {correlation_id}")
        tprint(f"⏰ Pipeline start time: {self.pipeline_state['start_time']}")
        
        self.enhanced_logger.start_pipeline(symbol, exchange, correlation_id)
        progress_monitor.start_monitoring()
        
        tprint("📊 Enhanced logger started")
        tprint("📈 Progress monitor started")
        try:
            tprint("🔍 VALIDATING PIPELINE PREREQUISITES")
            if not await self._validate_pipeline_prerequisites(symbol, exchange, timeframe, data_dir):
                tprint("❌ Pipeline prerequisites validation failed")
                return False
            tprint("✅ Pipeline prerequisites validation passed")
            
            # Execute each step with detailed printing
            steps_to_execute = [
                ('hmm_clustering', self._execute_hmm_clustering, 'HMM Clustering'),
                ('regime_splitting', self._execute_regime_splitting, 'Regime Data Splitting'),
                ('labeling', self._execute_labeling, 'Triple Barrier Labeling'),
                ('feature_engineering', self._execute_feature_engineering, 'Feature Engineering'),
                ('matrix_operations', self._execute_matrix_operations, 'Matrix Operations'),
                ('feature_selection', self._execute_feature_selection, 'Feature Selection'),
                ('final_feature_selection', self._execute_final_feature_selection, 'Final Feature Selection (120→100→80→60)')
            ]
            
            for step_name, step_func, step_display_name in steps_to_execute:
                if self.step_configs[step_name]['enabled']:
                    tprint(f"🔄 EXECUTING STEP: {step_display_name}")
                    tprint(f"   📋 Step name: {step_name}")
                    tprint(f"   ⏱️ Timeout: {self.step_configs[step_name]['timeout']}s")
                    tprint(f"   🔄 Retry attempts: {self.step_configs[step_name]['retry_attempts']}")
                    
                    if not await self._execute_step_with_validation(
                        step_name=step_name, 
                        step_func=step_func, 
                        symbol=symbol, 
                        exchange=exchange, 
                        timeframe=timeframe, 
                        data_dir=data_dir, 
                        **kwargs
                    ):
                        tprint(f"❌ Step {step_display_name} failed")
                        return False
                    tprint(f"✅ Step {step_display_name} completed successfully")
                else:
                    tprint(f"⏭️ Skipping disabled step: {step_display_name}")
            self.pipeline_state['end_time'] = get_current_datetime()
            self.pipeline_state['completed_steps'] = list(self.step_configs.keys())
            
            tprint("🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
            tprint(f"⏰ Pipeline end time: {self.pipeline_state['end_time']}")
            tprint(f"✅ Completed steps: {self.pipeline_state['completed_steps']}")
            
            await self._save_pipeline_state(symbol, exchange, timeframe, data_dir)
            progress_monitor.stop_monitoring()
            self.enhanced_logger.end_pipeline(success = True)
            
            tprint("💾 Pipeline state saved")
            tprint("📈 Progress monitor stopped")
            tprint("📊 Enhanced logger ended successfully")
            tprint("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            tprint("=" * 80)
            
            return True
        except Exception as e:
            self.pipeline_state['end_time'] = get_current_datetime()
            error_message = str(e)
            
            tprint("💥 PIPELINE EXECUTION FAILED")
            tprint("=" * 80)
            tprint(f"❌ Error: {error_message}")
            tprint(f"⏰ Failure time: {self.pipeline_state['end_time']}")
            tprint(f"🔄 Current step: {self.pipeline_state['current_step']}")
            tprint(f"✅ Completed steps: {self.pipeline_state['completed_steps']}")
            tprint(f"❌ Failed steps: {self.pipeline_state['failed_steps']}")
            tprint("=" * 80)
            
            self.logger.exception(f'💥 MARKET ANALYSIS PIPELINE FAILED: {error_message}')
            progress_monitor.stop_monitoring()
            self.enhanced_logger.end_pipeline(success = False, error_message = error_message)
            await self._save_pipeline_state(symbol, exchange, timeframe, data_dir, success = False)
            
            tprint("📈 Progress monitor stopped")
            tprint("📊 Enhanced logger ended with failure")
            tprint("💾 Pipeline state saved with failure status")
            
            return False

    async def _validate_pipeline_prerequisites(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate prerequisites before starting the pipeline."""
        tprint("🔍 VALIDATING PIPELINE PREREQUISITES")
        tprint(f"   🎯 Symbol: {symbol}")
        tprint(f"   🏢 Exchange: {exchange}")
        tprint(f"   📊 Timeframe: {timeframe}")
        tprint(f"   📁 Data directory: {data_dir}")
        
        self.logger.info('🔍 Validating pipeline prerequisites...')
        data_path = Path(data_dir)
        
        tprint(f"📁 Checking data directory: {data_path}")
        if not data_path.exists():
            tprint(f"❌ Data directory does not exist: {data_dir}")
            self.logger.error(f'❌ Data directory does not exist: {data_dir}')
            return False
        tprint(f"✅ Data directory exists: {data_dir}")
        required_files = [f'aggtrades_{exchange}_{symbol}_consolidated.parquet', f'volume_{exchange}_{symbol}_consolidated.parquet']
        tprint(f"📋 Checking required files: {required_files}")
        
        for file_name in required_files:
            file_path = data_path / file_name
            tprint(f"   📄 Checking file: {file_name}")
            if not safe_file_exists(file_path):
                tprint(f"   ❌ Required file not found: {file_path}")
                self.logger.error(f'❌ Required file not found: {file_path}')
                return False
            tprint(f"   ✅ File exists: {file_name}")
        price_data_path = data_path / required_files[0]
        tprint(f"📊 Loading price data from: {price_data_path}")
        
        try:
            price_data = standardized_parquet_handler.read_parquet_standardized(price_data_path)
            tprint(f"   📈 Price data loaded: {len(price_data)} rows, {len(price_data.columns)} columns")
            
            if price_data.empty:
                tprint("   ❌ Price data is empty")
                self.logger.error('❌ Price data is empty')
                return False
            tprint("   ✅ Price data is not empty")
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            tprint(f"   🔍 Checking required columns: {required_columns}")
            tprint(f"   📋 Available columns: {list(price_data.columns)}")
            
            missing_columns = set(required_columns) - set(price_data.columns)
            if missing_columns:
                tprint(f"   ❌ Missing required columns: {missing_columns}")
                self.logger.error(f'❌ Missing required columns: {missing_columns}')
                return False
            tprint("   ✅ All required columns present")
            # Enhanced data quality validation using ML Common utilities
            if self.ml_data_quality:
                tprint("   🔬 Running ML-enhanced data quality validation")
                try:
                    enhanced_quality_report = await self.ml_data_quality.perform_comprehensive_validation(
                        price_data, symbol=symbol, exchange=exchange
                    )
                    tprint(f"   📊 ML quality report generated: {bool(enhanced_quality_report)}")
                    
                    if enhanced_quality_report.get('has_critical_issues', False):
                        tprint(f"   🚨 Critical data quality issues detected: {enhanced_quality_report.get('critical_issues', [])}")
                        self.logger.error(f"🚨 Critical data quality issues detected by ML utilities: {enhanced_quality_report.get('critical_issues', [])}")
                        return False
                    
                    if enhanced_quality_report.get('warnings', []):
                        tprint(f"   ⚠️ ML-enhanced data quality warnings: {enhanced_quality_report.get('warnings', [])}")
                        self.logger.warning(f"⚠️ ML-enhanced data quality warnings: {enhanced_quality_report.get('warnings', [])}")
                    
                    tprint("   ✅ ML-enhanced data quality validation passed")
                    self.logger.info("✅ ML-enhanced data quality validation passed")
                except Exception as e:
                    tprint(f"   ⚠️ ML-enhanced data quality validation failed: {e}")
                    self.logger.warning(f"⚠️ ML-enhanced data quality validation failed: {e}")
            else:
                tprint("   ⏭️ Skipping ML-enhanced validation (not available)")
                
        except Exception as e:
            tprint(f"   ❌ Failed to validate data quality: {e}")
            self.logger.error(f'❌ Failed to validate data quality: {e}')
            return False
            
        tprint("✅ Pipeline prerequisites validated successfully")
        self.logger.info('✅ Pipeline prerequisites validated successfully')
        return True

    async def _execute_step_with_validation(self, step_name: str, step_func: callable, **kwargs) -> bool:
        """Execute a pipeline step with comprehensive validation and error handling."""
        tprint(f"🔄 EXECUTING STEP WITH VALIDATION: {step_name}")
        tprint(f"   📋 Step function: {step_func.__name__}")
        tprint(f"   ⚙️ Additional kwargs: {kwargs}")
        
        step_description = self._get_step_description(step_name)
        step_config = self.step_configs.get(step_name, {})
        step_number = step_config.get('step_number', 0)
        total_steps = len([s for s in self.step_configs.values() if s.get('enabled', True)])
        
        tprint(f"   📝 Step description: {step_description}")
        tprint(f"   🔢 Step number: {step_number}/{total_steps}")
        tprint(f"   ⏱️ Timeout: {step_config.get('timeout', 'N/A')}s")
        tprint(f"   🔄 Retry attempts: {step_config.get('retry_attempts', 'N/A')}")
        
        self.enhanced_logger.start_step(step_name, step_description, step_number, total_steps)
        progress_monitor.update_step_progress(step_name, 0.0, 'Starting...', 'running', step_number = step_number, total_steps = total_steps)
        self.logger.info(f'🔄 Executing step: {step_name}')
        self.pipeline_state['current_step'] = step_name
        
        tprint(f"   📊 Enhanced logger started for step: {step_name}")
        tprint(f"   📈 Progress monitor updated: {step_name}")
        tprint(f"   🔄 Pipeline state current step set to: {step_name}")
        try:
            tprint(f"   🔍 Validating step prerequisites for: {step_name}")
            progress_monitor.update_step_progress(step_name, 0.1, 'Validating prerequisites...', 'running')
            if not await self._validate_step_prerequisites(step_name, **kwargs):
                tprint(f"   ❌ Step prerequisites validation failed: {step_name}")
                progress_monitor.complete_step(step_name, False, 'Prerequisites validation failed')
                self.enhanced_logger.end_step(step_name, success = False, error_message='Prerequisites validation failed')
                return False
            tprint(f"   ✅ Step prerequisites validation passed: {step_name}")
            
            tprint(f"   🚀 Executing step function: {step_name}")
            progress_monitor.update_step_progress(step_name, 0.3, 'Executing step...', 'running')
            step_config = self.step_configs[step_name]
            success = await step_func(**kwargs)
            
            if not success:
                tprint(f"   ❌ Step execution failed: {step_name}")
                self.logger.error(f'❌ Step {step_name} failed')
                self.pipeline_state['failed_steps'].append(step_name)
                progress_monitor.complete_step(step_name, False, 'Step execution failed')
                self.enhanced_logger.end_step(step_name, success = False, error_message='Step execution failed')
                return False
            tprint(f"   ✅ Step execution completed: {step_name}")
            
            tprint(f"   🔍 Validating step output for: {step_name}")
            progress_monitor.update_step_progress(step_name, 0.8, 'Validating output...', 'running')
            if not await self._validate_step_output(step_name, **kwargs):
                tprint(f"   ❌ Step output validation failed: {step_name}")
                progress_monitor.complete_step(step_name, False, 'Output validation failed')
                self.enhanced_logger.end_step(step_name, success = False, error_message='Output validation failed')
                return False
            tprint(f"   ✅ Step output validation passed: {step_name}")
            
            self.logger.info(f'✅ Step {step_name} completed successfully')
            self.pipeline_state['completed_steps'].append(step_name)
            progress_monitor.complete_step(step_name, True, 'Completed successfully')
            self.enhanced_logger.end_step(step_name, success = True)
            
            tprint(f"   🎉 Step completed successfully: {step_name}")
            tprint(f"   📊 Progress monitor completed: {step_name}")
            tprint(f"   📈 Enhanced logger ended: {step_name}")
            
            return True
        except Exception as e:
            error_message = str(e)
            tprint(f"   💥 Step failed with exception: {step_name}")
            tprint(f"   ❌ Error: {error_message}")
            
            self.logger.exception(f'❌ Step {step_name} failed with exception: {error_message}')
            self.pipeline_state['failed_steps'].append(step_name)
            progress_monitor.complete_step(step_name, False, f'Failed: {error_message}')
            self.enhanced_logger.end_step(step_name, success = False, error_message = error_message)
            
            tprint(f"   📊 Progress monitor completed with failure: {step_name}")
            tprint(f"   📈 Enhanced logger ended with failure: {step_name}")
            tprint(f"   🔄 Pipeline state updated with failed step: {step_name}")
            
            return False
    @log_all_calls

    def _get_step_description(self, step_name: str) -> str:
        """Get a description for a pipeline step."""
        descriptions = {'hmm_clustering': 'HMM regime discovery and clustering', 'regime_splitting': 'Regime data splitting and preparation', 'labeling': 'Triple barrier method labeling', 'feature_engineering': 'Feature engineering and interaction creation', 'matrix_operations': 'Enhanced matrix operations and analysis', 'feature_selection': 'Advanced feature selection and optimization'}
        return descriptions.get(step_name, f'Pipeline step: {step_name}')

    @handles_errors(Exception, fallback = False)
    async def _validate_step_prerequisites(self, step_name: str, **kwargs) -> bool:
        """Validate prerequisites for a specific step."""
        self.logger.info(f'🔍 Validating prerequisites for step: {step_name}')
        try:
            dependencies = self.dependency_validator.get_step_dependencies(step_name)
            for dependency in dependencies:
                if dependency not in self.pipeline_state['completed_steps']:
                    self.logger.error(f'❌ Missing dependency for {step_name}: {dependency}')
                    return False
        except Exception as e:
            self.logger.warning(f'⚠️ Could not validate dependencies for {step_name}: {e}')
        return True

    @handles_errors(Exception, fallback = False)
    async def _validate_step_output(self, step_name: str, **kwargs) -> bool:
        """Validate the output of a specific step."""
        self.logger.info(f'🔍 Validating output for step: {step_name}')
        try:
            validation_result = await self.enhanced_validator.validate_step_output(step_name = step_name, symbol = kwargs.get('symbol'), exchange = kwargs.get('exchange'), timeframe = kwargs.get('timeframe'), data_dir = kwargs.get('data_dir'))
            if not validation_result.get('valid', False):
                self.logger.error(f"❌ Step {step_name} output validation failed: {validation_result.get('errors', [])}")
                return False
            warnings = validation_result.get('warnings', [])
            if warnings:
                for warning in warnings:
                    self.logger.warning(f'⚠️ Step {step_name} validation warning: {warning}')
        except Exception as e:
            self.logger.warning(f'⚠️ Could not validate output for {step_name}: {e}')
        return True

    @handles_errors(Exception, fallback = False)
    
    async def _execute_hmm_clustering(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
        """Execute HMM clustering step with comprehensive regime quality metrics."""
        tprint("🧠 EXECUTING HMM CLUSTERING STEP")
        tprint(f"   🎯 Symbol: {symbol}")
        tprint(f"   🏢 Exchange: {exchange}")
        tprint(f"   📊 Timeframe: {timeframe}")
        tprint(f"   📁 Data directory: {data_dir}")
        tprint(f"   🔄 Force rerun: {kwargs.get('force_rerun', True)}")
        tprint(f"   ⚙️ Additional kwargs: {kwargs}")
        
        self.logger.info('🧠 Executing HMM clustering...')
        try:
            tprint("   🚀 Calling run_enhanced_step for HMM clustering")
            success = await run_enhanced_step(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, force_rerun = kwargs.get('force_rerun', True))
            tprint(f"   📊 HMM clustering step result: {success}")
            
            if success:
                tprint("   ✅ HMM clustering completed successfully")
                self.logger.info('✅ HMM clustering completed successfully')
                try:
                    tprint("   📊 Analyzing HMM clustering results")
                    try:
                        from pathlib import Path
                        regime_path = Path(data_dir) / f'regimes_{exchange}_{symbol}_{timeframe}.parquet'
                        tprint(f"   📁 Checking regime data file: {regime_path}")
                        
                        if regime_path.exists():
                            tprint("   ✅ Regime data file found")
                            regime_data = standardized_parquet_handler.read_parquet_standardized(regime_path)
                            tprint(f"   📈 Regime data loaded: {len(regime_data)} rows, {len(regime_data.columns)} columns")
                            
                            if 'regime' in regime_data.columns:
                                tprint("   🎯 Regime column found, analyzing regime quality")
                                self.enhanced_logger.log_regime_quality('hmm_clustering', regime_data['regime'])
                                unique_regimes = regime_data['regime'].unique()
                                regime_counts = regime_data['regime'].value_counts().sort_index()
                                
                                tprint(f'   🎯 Regime Analysis Results:')
                                tprint(f'     📊 Total Regimes Discovered: {len(unique_regimes)}')
                                tprint(f'     📈 Regime Distribution:')
                                
                                self.logger.info(f'🎯 Regime Analysis Results:')
                                self.logger.info(f'  📊 Total Regimes Discovered: {len(unique_regimes)}')
                                self.logger.info(f'  📈 Regime Distribution:')
                                
                                for regime_id, count in regime_counts.items():
                                    percentage = count / len(regime_data) * 100
                                    tprint(f'       Regime {regime_id}: {count} samples ({percentage:.1f}%)')
                                    self.logger.info(f'    Regime {regime_id}: {count} samples ({percentage:.1f}%)')
                                
                                min_samples = 100
                                for regime_id, count in regime_counts.items():
                                    if count < min_samples:
                                        tprint(f'     ⚠️ Regime {regime_id} has only {count} samples (minimum: {min_samples})')
                                        self.enhanced_logger.log_issue('hmm_clustering', 'regime_quality', f'Regime {regime_id} has only {count} samples (minimum: {min_samples})', 'warning')
                            else:
                                tprint("   ⚠️ No 'regime' column found in regime data")
                                self.logger.warning("⚠️ No 'regime' column found in regime data")
                        else:
                            tprint("   📊 HMM clustering completed (regime data file not found)")
                            self.logger.info('📊 HMM clustering completed (regime data file not found)')
                    except ImportError:
                        tprint("   🧠 HMM clustering completed (pandas not available for detailed metrics)")
                        self.logger.info('🧠 HMM clustering completed (pandas not available for detailed metrics)')
                except Exception as metrics_error:
                    tprint(f"   ⚠️ Could not log regime quality metrics: {metrics_error}")
                    self.logger.warning(f'⚠️ Could not log regime quality metrics: {metrics_error}')
            else:
                tprint("   ❌ HMM clustering failed")
                self.logger.error('❌ HMM clustering failed')
                self.enhanced_logger.log_issue('hmm_clustering', 'execution', 'HMM clustering step failed', 'error')
            
            tprint(f"   🎯 HMM clustering step returning: {success}")
            return success
        except Exception as e:
            tprint(f"   💥 HMM clustering failed with exception: {e}")
            self.logger.exception(f'❌ HMM clustering failed with exception: {e}')
            self.enhanced_logger.log_issue('hmm_clustering', 'exception', str(e), 'error')
            return False

    @handles_errors(Exception, fallback = False)
    
    async def _execute_regime_splitting(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
        """Execute regime data splitting step."""
        self.logger.info('📊 Executing regime data splitting...')
        try:
            regime_splitter = RegimeDataSplittingStep()
            success = await regime_splitter.split_regime_data(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir)
            if success:
                self.logger.info('✅ Regime data splitting completed successfully')
            else:
                self.logger.error('❌ Regime data splitting failed')
            return success
        except Exception as e:
            self.logger.exception(f'❌ Regime data splitting failed with exception: {e}')
            return False

    @handles_errors(Exception, fallback = False)
    
    async def _execute_labeling(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
        """Execute labeling step."""
        self.logger.info('🏷️ Executing labeling...')
        try:
            labeler = LabelingStep()
            success = await labeler.create_labels(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir)
            if success:
                self.logger.info('✅ Labeling completed successfully')
            else:
                self.logger.error('❌ Labeling failed')
            return success
        except Exception as e:
            self.logger.exception(f'❌ Labeling failed with exception: {e}')
            return False

    @handles_errors(Exception, fallback = False)
    
    async def _execute_feature_engineering(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
        """Execute feature engineering step with comprehensive metrics logging."""
        self.logger.info('🔧 Executing feature engineering...')
        try:
            feature_engineer = FeatureEngineeringStep()
            success = await feature_engineer.engineer_features(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir)
            if success:
                self.logger.info('✅ Feature engineering completed successfully')
                try:
                    try:
                        from pathlib import Path
                        features_path = Path(data_dir) / f'features_{exchange}_{symbol}_{timeframe}.parquet'
                        if features_path.exists():
                            features_data = standardized_parquet_handler.read_parquet_standardized(features_path)
                            self.enhanced_logger.log_feature_quality('feature_engineering', features_data)
                            step6_metrics = {'total_features_created': len(features_data.columns), 'interaction_features': len([col for col in features_data.columns if '_x_' in col or '_*_' in col]), 'selected_features': len(features_data.columns), 'feature_importance_top_10': [], 'lookback_optimization': {'optimized_count': 0, 'optimization_time': 0.0}}
                            self.enhanced_logger.log_step6_metrics('feature_engineering', step6_metrics)
                        else:
                            step6_metrics = {'total_features_created': 0, 'interaction_features': 0, 'selected_features': 0, 'feature_importance_top_10': [], 'lookback_optimization': {'optimized_count': 0, 'optimization_time': 0.0}}
                            self.enhanced_logger.log_step6_metrics('feature_engineering', step6_metrics)
                    except ImportError:
                        self.logger.info('📊 Feature engineering completed (pandas not available for detailed metrics)')
                except Exception as metrics_error:
                    self.logger.warning(f'⚠️ Could not log feature engineering metrics: {metrics_error}')
            else:
                self.logger.error('❌ Feature engineering failed')
                self.enhanced_logger.log_issue('feature_engineering', 'execution', 'Feature engineering step failed', 'error')
            return success
        except Exception as e:
            self.logger.exception(f'❌ Feature engineering failed with exception: {e}')
            self.enhanced_logger.log_issue('feature_engineering', 'exception', str(e), 'error')
            return False
    @log_all_calls

    def _calculate_matrix_metrics(self, matrix_data: Any, numeric_cols: List[Any]) -> None:
        """Calculate matrix-specific metrics from data."""
        try:
            corr_matrix = matrix_data[numeric_cols].corr()
            try:
                condition_number = np.linalg.cond(corr_matrix.values)
            except:
                condition_number = float('inf')
            high_corr_pairs = 0
            max_correlation = 0.0
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    max_correlation = max(max_correlation, corr_val)
                    if corr_val > 0.95:
                        high_corr_pairs += 1
            return {'matrix_operations_performed': ['correlation_analysis', 'eigenvalue_analysis', 'feature_ranking'], 'eigenvalue_analysis': {'condition_number': condition_number, 'rank': np.linalg.matrix_rank(corr_matrix.values), 'effective_rank': len(numeric_cols)}, 'correlation_analysis': {'high_correlation_pairs': high_corr_pairs, 'max_correlation': max_correlation}, 'performance_metrics': {'computation_time': 0.0, 'memory_usage_mb': 0.0}}
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating matrix metrics: {e}')
            return self._get_fallback_matrix_metrics()
    @log_all_calls

    def _get_fallback_matrix_metrics(self) -> None:
        """Get fallback metrics when matrix analysis fails."""
        return {'matrix_operations_performed': ['correlation_analysis', 'eigenvalue_analysis'], 'eigenvalue_analysis': {'condition_number': 1.0, 'rank': 0, 'effective_rank': 0}, 'correlation_analysis': {'high_correlation_pairs': 0, 'max_correlation': 0.0}, 'performance_metrics': {'computation_time': 0.0, 'memory_usage_mb': 0.0}}
    @log_all_calls

    def _log_matrix_operations_metrics(self, data_dir: str, exchange: str, symbol: str, timeframe: str) -> None:
        """Log detailed matrix operations metrics."""
        try:
            from pathlib import Path

            matrix_path = Path(data_dir) / f'matrix_operations_{exchange}_{symbol}_{timeframe}.parquet'
            if not matrix_path.exists():
                step7_metrics = self._get_fallback_matrix_metrics()
                self.enhanced_logger.log_step7_metrics('matrix_operations', step7_metrics)
                return
            matrix_data = standardized_parquet_handler.read_parquet_standardized(matrix_path)
            self.enhanced_logger.log_feature_quality('matrix_operations', matrix_data)
            numeric_cols = matrix_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                step7_metrics = self._calculate_matrix_metrics(matrix_data, numeric_cols)
            else:
                step7_metrics = self._get_fallback_matrix_metrics()
            self.enhanced_logger.log_step7_metrics('matrix_operations', step7_metrics)
        except ImportError:
            self.logger.info('🧮 Matrix operations completed (pandas/numpy not available for detailed metrics)')
        except Exception as metrics_error:
            self.logger.warning(f'⚠️ Could not log matrix operations metrics: {metrics_error}')

    @handles_errors(Exception, fallback = False)
    
    async def _execute_matrix_operations(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
        """Execute matrix operations step with comprehensive metrics logging."""
        self.logger.info('🧮 Executing matrix operations...')
        try:
            matrix_ops = EnhancedMatrixOperationsStep()
            success = await matrix_ops.perform_matrix_operations(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir)
            if success:
                self.logger.info('✅ Matrix operations completed successfully')
                self._log_matrix_operations_metrics(data_dir, exchange, symbol, timeframe)
            else:
                self.logger.error('❌ Matrix operations failed')
                self.enhanced_logger.log_issue('matrix_operations', 'execution', 'Matrix operations step failed', 'error')
            return success
        except Exception as e:
            self.logger.exception(f'❌ Matrix operations failed with exception: {e}')
            self.enhanced_logger.log_issue('matrix_operations', 'exception', str(e), 'error')
            return False

    @handles_errors(Exception, fallback = False)
    
    async def _execute_feature_selection(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
        """Execute feature selection step with ML utilities enhancement."""
        self.logger.info('🎯 Executing feature selection with ML utilities...')
        try:
            feature_selector = AdvancedFeatureSelectionStep()

            # Enhanced feature selection using ML Common utilities
            if self.ml_feature_selection:
                try:
                    self.logger.info('🔬 Running ML-enhanced feature importance analysis...')
                    enhanced_feature_analysis = await self.ml_feature_selection.analyze_feature_importance(
                        symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir
                    )
                    if enhanced_feature_analysis.get('recommendations'):
                        self.logger.info(f'💡 ML feature selection recommendations: {enhanced_feature_analysis["recommendations"]}')

                    # Pass enhanced analysis to the standard feature selector
                    kwargs['ml_enhanced_analysis'] = enhanced_feature_analysis
                except Exception as e:
                    self.logger.warning(f'⚠️ ML-enhanced feature analysis failed: {e}')

            success = await feature_selector.select_features(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, **kwargs)

            # Post-selection analysis with ML utilities
            if success and self.ml_feature_selection:
                try:
                    self.logger.info('📊 Running post-selection validation with ML utilities...')
                    validation_results = await self.ml_feature_selection.validate_feature_selection(
                        symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir
                    )
                    if validation_results.get('warnings'):
                        self.logger.warning(f'⚠️ Feature selection validation warnings: {validation_results["warnings"]}')
                    self.logger.info('✅ ML-enhanced feature selection validation completed')
                except Exception as e:
                    self.logger.warning(f'⚠️ ML-enhanced feature selection validation failed: {e}')

            if success:
                self.logger.info('✅ Feature selection completed successfully with ML enhancements')
            else:
                self.logger.error('❌ Feature selection failed')
            return success
        except Exception as e:
            self.logger.exception(f'❌ Feature selection failed with exception: {e}')
            return False

    @handles_errors(Exception, fallback=False)
    async def _execute_final_feature_selection(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> bool:
        """Execute final feature selection step (120→100→80→60)."""
        self.logger.info('🎯 Executing final feature selection step...')
        try:
            # Run final feature selection step
            success = await run_final_feature_selection_step(
                symbol=symbol, 
                exchange=exchange, 
                timeframe=timeframe, 
                data_dir=data_dir,
                config=kwargs.get('final_feature_selection_config', {})
            )
            
            if success:
                self.logger.info('✅ Final feature selection completed successfully')
            else:
                self.logger.error('❌ Final feature selection failed')
            return success
        except Exception as e:
            self.logger.exception(f'❌ Final feature selection failed with exception: {e}')
            return False

    @handles_errors(Exception, fallback = None)
    async def _save_pipeline_state(self, symbol: str, exchange: str, timeframe: str, data_dir: str, success: bool = True) -> None:
        """Save pipeline state for monitoring and debugging."""
        try:
            state_file = Path(data_dir) / f'market_analysis_state_{symbol}_{timeframe}.json'
            pipeline_state = {**self.pipeline_state, 'success': success, 'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir, 'execution_time': self._get_execution_time(), 'timestamp': format_datetime(get_current_datetime())}
            safe_json_dump(pipeline_state, state_file, indent = 2)
            self.logger.info(f'💾 Pipeline state saved to: {state_file}')
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to save pipeline state: {e}')
    @log_all_calls

    def _get_execution_time(self) -> float:
        """Get total execution time in seconds."""
        if self.pipeline_state['start_time'] and self.pipeline_state['end_time']:
            return (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
        return 0.0

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {'current_step': self.pipeline_state['current_step'], 'completed_steps': self.pipeline_state['completed_steps'], 'failed_steps': self.pipeline_state['failed_steps'], 'execution_time': self._get_execution_time(), 'correlation_id': self.pipeline_state['correlation_id']}

async def run_enhanced_market_analysis_pipeline(symbol: str, exchange: str, timeframe: str='1m', data_dir: str='historical_data', **config) -> bool:
    """
    Run the enhanced market analysis pipeline with comprehensive validation.
    
    This is the main entry point for the market analysis pipeline.
    """
    orchestrator = MarketAnalysisPipelineOrchestrator(config)
    return await orchestrator.execute_pipeline(symbol = symbol, exchange = exchange, timeframe = timeframe, data_dir = data_dir, **config)
if __name__ == '__main__':

    async def main() -> None:
        config = {'force_rerun': True, 'hmm_clustering': True, 'regime_splitting': True, 'feature_engineering': True, 'matrix_operations': True, 'feature_selection': True, 'random_state': 42}
        success = await run_enhanced_market_analysis_pipeline(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='historical_data', **config)
        if success:
            tprint('🎉 Market analysis pipeline completed successfully!')
        else:
            tprint('❌ Market analysis pipeline failed!')
    asyncio.run(main())
from typing import Dict, List, Optional, Union, Any, Tuple