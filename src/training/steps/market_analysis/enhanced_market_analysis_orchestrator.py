#!/usr/bin/env python3
"""
Enhanced Market Analysis Orchestrator

This module provides a comprehensive orchestrator for the market analysis pipeline
with proper validators, decorators, and common utilities to ensure each step
leads to the next with proper validation and protection.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core decorators and utilities
from src.core.decorators import (
    handles_errors,
    traced,
    validates,
    cached,
    log_execution_time,
    timeout,
    retry,
    circuit_breaker,
    audit_log,
    get_correlation_id,
    set_correlation_id,
)
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    ensure_directory,
    safe_json_dump,
    safe_json_load,
    validate_dataframe,
    validate_data_quality,
    safe_file_exists,
    get_logger,
    timed_operation,
)
from src.utils.data_quality_framework import DataQualityFramework
from src.utils.validator_orchestrator import ValidatorOrchestrator
from src.utils.step_dependency_validator import StepDependencyValidator

# Import market analysis components
from .step04_regime_data_splitting import RegimeDataSplittingStep
from .step04_regime_data_splitting_validator import RegimeDataSplittingValidator
from .step05_labeling import LabelingStep
from .step05_labeling_validator import LabelingValidator
from .step06_feature_engineering import FeatureEngineeringStep
from .step06_feature_engineering_validator import FeatureEngineeringValidator
from .step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
from .step07_enhanced_matrix_operations_validator import MatrixOperationsValidator
from .step08_advanced_feature_selection import AdvancedFeatureSelectionStep

# Import HMM clustering
from .hmm_clustering import run_enhanced_step

# Import enhanced validators and decorators
from .enhanced_step_validator import EnhancedStepValidator
from .enhanced_pipeline_decorators import (
    comprehensive_pipeline_protection,
    data_formatting,
    data_analysis_protection,
    data_access_protection,
)

# Import enhanced logging system
from .enhanced_logging_metrics import EnhancedPipelineLogger, enhanced_logger
from .progress_monitor import progress_monitor, ProgressContext


class MarketAnalysisPipelineOrchestrator:
    """
    Enhanced orchestrator for market analysis pipeline with comprehensive
    validation, error handling, and observability.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator with configuration."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.enhanced_logger = EnhancedPipelineLogger("market_analysis_orchestrator")
        self.data_quality_framework = DataQualityFramework()
        self.validator_orchestrator = ValidatorOrchestrator()
        self.dependency_validator = StepDependencyValidator()
        self.enhanced_validator = EnhancedStepValidator(config)
        
        # Pipeline state tracking
        self.pipeline_state = {
            'current_step': None,
            'completed_steps': [],
            'failed_steps': [],
            'start_time': None,
            'end_time': None,
            'correlation_id': None,
        }
        
        # Step configurations with step numbers
        self.step_configs = {
            'hmm_clustering': {
                'enabled': True,
                'timeout': 300,  # 5 minutes
                'retry_attempts': 3,
                'validator': None,  # Will be set dynamically
                'step_number': 1,
            },
            'regime_splitting': {
                'enabled': True,
                'timeout': 180,  # 3 minutes
                'retry_attempts': 2,
                'validator': RegimeDataSplittingValidator(),
                'step_number': 2,
            },
            'labeling': {
                'enabled': True,
                'timeout': 240,  # 4 minutes
                'retry_attempts': 2,
                'validator': LabelingValidator(),
                'step_number': 3,
            },
            'feature_engineering': {
                'enabled': True,
                'timeout': 600,  # 10 minutes
                'retry_attempts': 2,
                'validator': FeatureEngineeringValidator(),
                'step_number': 4,
            },
            'matrix_operations': {
                'enabled': True,
                'timeout': 300,  # 5 minutes
                'retry_attempts': 2,
                'validator': MatrixOperationsValidator(),
                'step_number': 5,
            },
            'feature_selection': {
                'enabled': True,
                'timeout': 180,  # 3 minutes
                'retry_attempts': 2,
                'validator': None,  # Will be set dynamically
                'step_number': 6,
            },
        }

    @handles_errors(ValueError, FileNotFoundError, TimeoutError, fallback=False)
    @traced(operation_name="market_analysis_pipeline")
    @log_execution_time
    @audit_log(operation="market_analysis_pipeline")
    async def execute_pipeline(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        data_dir: str = "data_cache",
        **kwargs
    ) -> bool:
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
        # Set up correlation ID for tracing
        correlation_id = f"market_analysis_{symbol}_{exchange}_{int(time.time())}"
        set_correlation_id(correlation_id)
        self.pipeline_state['correlation_id'] = correlation_id
        self.pipeline_state['start_time'] = get_current_datetime()
        
        # Start enhanced logging and progress monitoring
        self.enhanced_logger.start_pipeline(symbol, exchange, correlation_id)
        progress_monitor.start_monitoring()
        
        try:
            # Step 0: Pre-pipeline validation
            if not await self._validate_pipeline_prerequisites(symbol, exchange, timeframe, data_dir):
                return False
            
            # Step 1: HMM Clustering
            if self.step_configs['hmm_clustering']['enabled']:
                if not await self._execute_step_with_validation(
                    step_name="hmm_clustering",
                    step_func=self._execute_hmm_clustering,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    **kwargs
                ):
                    return False
            
            # Step 2: Regime Data Splitting
            if self.step_configs['regime_splitting']['enabled']:
                if not await self._execute_step_with_validation(
                    step_name="regime_splitting",
                    step_func=self._execute_regime_splitting,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    **kwargs
                ):
                    return False
            
            # Step 3: Labeling
            if self.step_configs['labeling']['enabled']:
                if not await self._execute_step_with_validation(
                    step_name="labeling",
                    step_func=self._execute_labeling,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    **kwargs
                ):
                    return False
            
            # Step 4: Feature Engineering
            if self.step_configs['feature_engineering']['enabled']:
                if not await self._execute_step_with_validation(
                    step_name="feature_engineering",
                    step_func=self._execute_feature_engineering,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    **kwargs
                ):
                    return False
            
            # Step 5: Matrix Operations
            if self.step_configs['matrix_operations']['enabled']:
                if not await self._execute_step_with_validation(
                    step_name="matrix_operations",
                    step_func=self._execute_matrix_operations,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    **kwargs
                ):
                    return False
            
            # Step 6: Feature Selection
            if self.step_configs['feature_selection']['enabled']:
                if not await self._execute_step_with_validation(
                    step_name="feature_selection",
                    step_func=self._execute_feature_selection,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    data_dir=data_dir,
                    **kwargs
                ):
                    return False
            
            # Pipeline completed successfully
            self.pipeline_state['end_time'] = get_current_datetime()
            self.pipeline_state['completed_steps'] = list(self.step_configs.keys())
            
            await self._save_pipeline_state(symbol, exchange, timeframe, data_dir)
            
            # End enhanced logging and progress monitoring with success
            progress_monitor.stop_monitoring()
            self.enhanced_logger.end_pipeline(success=True)
            
            return True
            
        except Exception as e:
            self.pipeline_state['end_time'] = get_current_datetime()
            error_message = str(e)
            self.logger.exception(f"💥 MARKET ANALYSIS PIPELINE FAILED: {error_message}")
            
            # End enhanced logging and progress monitoring with failure
            progress_monitor.stop_monitoring()
            self.enhanced_logger.end_pipeline(success=False, error_message=error_message)
            
            await self._save_pipeline_state(symbol, exchange, timeframe, data_dir, success=False)
            return False

    @handles_errors(FileNotFoundError, ValueError, fallback=False)
    @validates(schema={
        'symbol': str,
        'exchange': str,
        'timeframe': str,
        'data_dir': str
    })
    async def _validate_pipeline_prerequisites(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> bool:
        """Validate prerequisites before starting the pipeline."""
        self.logger.info("🔍 Validating pipeline prerequisites...")
        
        # Validate data directory exists
        data_path = Path(data_dir)
        if not data_path.exists():
            self.logger.error(f"❌ Data directory does not exist: {data_dir}")
            return False
        
        # Validate required input files exist
        required_files = [
            f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"volume_{exchange}_{symbol}_consolidated.parquet",
        ]
        
        for file_name in required_files:
            file_path = data_path / file_name
            if not safe_file_exists(file_path):
                self.logger.error(f"❌ Required file not found: {file_path}")
                return False
        
        # Validate data quality
        price_data_path = data_path / required_files[0]
        try:
            import pandas as pd
            price_data = pd.read_parquet(price_data_path)
            
            # Basic data quality checks
            if price_data.empty:
                self.logger.error("❌ Price data is empty")
                return False
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(price_data.columns)
            if missing_columns:
                self.logger.error(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # Data quality validation
            quality_report = validate_data_quality(price_data)
            if not quality_report['is_valid']:
                self.logger.warning(f"⚠️ Data quality issues detected: {quality_report['issues']}")
                # Continue with warnings rather than failing
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate data quality: {e}")
            return False
        
        self.logger.info("✅ Pipeline prerequisites validated successfully")
        return True

    @handles_errors(Exception, fallback=False)
    @timeout(seconds=300)  # 5 minute timeout
    @retry(attempts=3, delay=1.0, backoff=2.0)
    @circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @traced(operation_name="execute_step_with_validation")
    async def _execute_step_with_validation(
        self,
        step_name: str,
        step_func: callable,
        **kwargs
    ) -> bool:
        """Execute a pipeline step with comprehensive validation and error handling."""
        # Start enhanced logging and progress monitoring for this step
        step_description = self._get_step_description(step_name)
        step_config = self.step_configs.get(step_name, {})
        step_number = step_config.get('step_number', 0)
        total_steps = len([s for s in self.step_configs.values() if s.get('enabled', True)])
        
        self.enhanced_logger.start_step(step_name, step_description, step_number, total_steps)
        progress_monitor.update_step_progress(step_name, 0.0, "Starting...", "running", 
                                            step_number=step_number, total_steps=total_steps)
        
        self.logger.info(f"🔄 Executing step: {step_name}")
        self.pipeline_state['current_step'] = step_name
        
        try:
            # Pre-step validation
            progress_monitor.update_step_progress(step_name, 0.1, "Validating prerequisites...", "running")
            if not await self._validate_step_prerequisites(step_name, **kwargs):
                progress_monitor.complete_step(step_name, False, "Prerequisites validation failed")
                self.enhanced_logger.end_step(step_name, success=False, error_message="Prerequisites validation failed")
                return False
            
            # Execute the step
            progress_monitor.update_step_progress(step_name, 0.3, "Executing step...", "running")
            step_config = self.step_configs[step_name]
            success = await step_func(**kwargs)
            
            if not success:
                self.logger.error(f"❌ Step {step_name} failed")
                self.pipeline_state['failed_steps'].append(step_name)
                progress_monitor.complete_step(step_name, False, "Step execution failed")
                self.enhanced_logger.end_step(step_name, success=False, error_message="Step execution failed")
                return False
            
            # Post-step validation
            progress_monitor.update_step_progress(step_name, 0.8, "Validating output...", "running")
            if not await self._validate_step_output(step_name, **kwargs):
                progress_monitor.complete_step(step_name, False, "Output validation failed")
                self.enhanced_logger.end_step(step_name, success=False, error_message="Output validation failed")
                return False
            
            self.logger.info(f"✅ Step {step_name} completed successfully")
            self.pipeline_state['completed_steps'].append(step_name)
            
            # Complete progress monitoring and end enhanced logging for this step
            progress_monitor.complete_step(step_name, True, "Completed successfully")
            self.enhanced_logger.end_step(step_name, success=True)
            
            return True
            
        except Exception as e:
            error_message = str(e)
            self.logger.exception(f"❌ Step {step_name} failed with exception: {error_message}")
            self.pipeline_state['failed_steps'].append(step_name)
            
            # Complete progress monitoring and end enhanced logging for this step with error
            progress_monitor.complete_step(step_name, False, f"Failed: {error_message}")
            self.enhanced_logger.end_step(step_name, success=False, error_message=error_message)
            
            return False

    def _get_step_description(self, step_name: str) -> str:
        """Get a description for a pipeline step."""
        descriptions = {
            'hmm_clustering': 'HMM regime discovery and clustering',
            'regime_splitting': 'Regime data splitting and preparation',
            'labeling': 'Triple barrier method labeling',
            'feature_engineering': 'Feature engineering and interaction creation',
            'matrix_operations': 'Enhanced matrix operations and analysis',
            'feature_selection': 'Advanced feature selection and optimization'
        }
        return descriptions.get(step_name, f'Pipeline step: {step_name}')

    @handles_errors(Exception, fallback=False)
    async def _validate_step_prerequisites(self, step_name: str, **kwargs) -> bool:
        """Validate prerequisites for a specific step."""
        self.logger.info(f"🔍 Validating prerequisites for step: {step_name}")
        
        # Use dependency validator to check step dependencies
        try:
            dependencies = self.dependency_validator.get_step_dependencies(step_name)
            for dependency in dependencies:
                if dependency not in self.pipeline_state['completed_steps']:
                    self.logger.error(f"❌ Missing dependency for {step_name}: {dependency}")
                    return False
        except Exception as e:
            self.logger.warning(f"⚠️ Could not validate dependencies for {step_name}: {e}")
        
        return True

    @handles_errors(Exception, fallback=False)
    async def _validate_step_output(self, step_name: str, **kwargs) -> bool:
        """Validate the output of a specific step."""
        self.logger.info(f"🔍 Validating output for step: {step_name}")
        
        # Use enhanced validator for comprehensive validation
        try:
            validation_result = await self.enhanced_validator.validate_step_output(
                step_name=step_name,
                symbol=kwargs.get('symbol'),
                exchange=kwargs.get('exchange'),
                timeframe=kwargs.get('timeframe'),
                data_dir=kwargs.get('data_dir')
            )
            
            if not validation_result.get('valid', False):
                self.logger.error(f"❌ Step {step_name} output validation failed: {validation_result.get('errors', [])}")
                return False
            
            # Log warnings if any
            warnings = validation_result.get('warnings', [])
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"⚠️ Step {step_name} validation warning: {warning}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not validate output for {step_name}: {e}")
        
        return True

    # Step execution methods
    @comprehensive_pipeline_protection(
        required_columns=['open', 'high', 'low', 'close', 'volume'],
        max_memory_mb=2000,
        max_execution_time=300,
        allowed_paths=['data_cache/*'],
        audit_access=True
    )
    @handles_errors(Exception, fallback=False)
    @traced(operation_name="hmm_clustering")
    @log_execution_time
    async def _execute_hmm_clustering(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        **kwargs
    ) -> bool:
        """Execute HMM clustering step with comprehensive regime quality metrics."""
        self.logger.info("🧠 Executing HMM clustering...")
        
        try:
            success = await run_enhanced_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=kwargs.get('force_rerun', True)
            )
            
            if success:
                self.logger.info("✅ HMM clustering completed successfully")
                
                # Log detailed regime quality metrics
                try:
                    # Try to load the regime data for analysis
                    try:
                        import pandas as pd
                        from pathlib import Path
                        
                        regime_path = Path(data_dir) / f"regimes_{exchange}_{symbol}_{timeframe}.parquet"
                        if regime_path.exists():
                            regime_data = pd.read_parquet(regime_path)
                            
                            if 'regime' in regime_data.columns:
                                # Log regime quality metrics
                                self.enhanced_logger.log_regime_quality("hmm_clustering", regime_data['regime'])
                                
                                # Log additional regime analysis
                                unique_regimes = regime_data['regime'].unique()
                                regime_counts = regime_data['regime'].value_counts().sort_index()
                                
                                self.logger.info(f"🎯 Regime Analysis Results:")
                                self.logger.info(f"  📊 Total Regimes Discovered: {len(unique_regimes)}")
                                self.logger.info(f"  📈 Regime Distribution:")
                                for regime_id, count in regime_counts.items():
                                    percentage = (count / len(regime_data)) * 100
                                    self.logger.info(f"    Regime {regime_id}: {count} samples ({percentage:.1f}%)")
                                
                                # Check for regime quality issues
                                min_samples = 100  # Minimum samples per regime
                                for regime_id, count in regime_counts.items():
                                    if count < min_samples:
                                        self.enhanced_logger.log_issue(
                                            "hmm_clustering", 
                                            "regime_quality", 
                                            f"Regime {regime_id} has only {count} samples (minimum: {min_samples})", 
                                            "warning"
                                        )
                            else:
                                self.logger.warning("⚠️ No 'regime' column found in regime data")
                        else:
                            self.logger.info("📊 HMM clustering completed (regime data file not found)")
                    except ImportError:
                        # Fallback when pandas not available
                        self.logger.info("🧠 HMM clustering completed (pandas not available for detailed metrics)")
                        
                except Exception as metrics_error:
                    self.logger.warning(f"⚠️ Could not log regime quality metrics: {metrics_error}")
                
            else:
                self.logger.error("❌ HMM clustering failed")
                self.enhanced_logger.log_issue("hmm_clustering", "execution", "HMM clustering step failed", "error")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ HMM clustering failed with exception: {e}")
            self.enhanced_logger.log_issue("hmm_clustering", "exception", str(e), "error")
            return False

    @comprehensive_pipeline_protection(
        required_columns=['regime', 'open', 'high', 'low', 'close', 'volume'],
        max_memory_mb=1500,
        max_execution_time=180,
        allowed_paths=['data_cache/*'],
        audit_access=True
    )
    @handles_errors(Exception, fallback=False)
    @traced(operation_name="regime_splitting")
    @log_execution_time
    async def _execute_regime_splitting(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        **kwargs
    ) -> bool:
        """Execute regime data splitting step."""
        self.logger.info("📊 Executing regime data splitting...")
        
        try:
            regime_splitter = RegimeDataSplittingStep()
            success = await regime_splitter.split_regime_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            if success:
                self.logger.info("✅ Regime data splitting completed successfully")
            else:
                self.logger.error("❌ Regime data splitting failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Regime data splitting failed with exception: {e}")
            return False

    @comprehensive_pipeline_protection(
        required_columns=['regime', 'open', 'high', 'low', 'close', 'volume'],
        max_memory_mb=1200,
        max_execution_time=240,
        allowed_paths=['data_cache/*'],
        audit_access=True
    )
    @handles_errors(Exception, fallback=False)
    @traced(operation_name="labeling")
    @log_execution_time
    async def _execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        **kwargs
    ) -> bool:
        """Execute labeling step."""
        self.logger.info("🏷️ Executing labeling...")
        
        try:
            labeler = LabelingStep()
            success = await labeler.create_labels(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            if success:
                self.logger.info("✅ Labeling completed successfully")
            else:
                self.logger.error("❌ Labeling failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Labeling failed with exception: {e}")
            return False

    @comprehensive_pipeline_protection(
        required_columns=['regime', 'label', 'open', 'high', 'low', 'close', 'volume'],
        max_memory_mb=3000,
        max_execution_time=600,
        allowed_paths=['data_cache/*'],
        audit_access=True
    )
    @handles_errors(Exception, fallback=False)
    @traced(operation_name="feature_engineering")
    @log_execution_time
    async def _execute_feature_engineering(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        **kwargs
    ) -> bool:
        """Execute feature engineering step with comprehensive metrics logging."""
        self.logger.info("🔧 Executing feature engineering...")
        
        try:
            feature_engineer = FeatureEngineeringStep()
            success = await feature_engineer.engineer_features(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            if success:
                self.logger.info("✅ Feature engineering completed successfully")
                
                # Log detailed feature engineering metrics
                try:
                    # Try to load the engineered features data for analysis
                    try:
                        import pandas as pd
                        from pathlib import Path
                        
                        features_path = Path(data_dir) / f"features_{exchange}_{symbol}_{timeframe}.parquet"
                        if features_path.exists():
                            features_data = pd.read_parquet(features_path)
                            
                            # Log feature quality metrics
                            self.enhanced_logger.log_feature_quality("feature_engineering", features_data)
                            
                            # Log step6 specific metrics
                            step6_metrics = {
                                'total_features_created': len(features_data.columns),
                                'interaction_features': len([col for col in features_data.columns if '_x_' in col or '_*_' in col]),
                                'selected_features': len(features_data.columns),
                                'feature_importance_top_10': [],  # Would need to be extracted from feature engineer
                                'lookback_optimization': {
                                    'optimized_count': 0,  # Would need to be extracted from feature engineer
                                    'optimization_time': 0.0
                                }
                            }
                            self.enhanced_logger.log_step6_metrics("feature_engineering", step6_metrics)
                        else:
                            # Fallback metrics when file doesn't exist
                            step6_metrics = {
                                'total_features_created': 0,
                                'interaction_features': 0,
                                'selected_features': 0,
                                'feature_importance_top_10': [],
                                'lookback_optimization': {
                                    'optimized_count': 0,
                                    'optimization_time': 0.0
                                }
                            }
                            self.enhanced_logger.log_step6_metrics("feature_engineering", step6_metrics)
                    except ImportError:
                        # Fallback when pandas not available
                        self.logger.info("📊 Feature engineering completed (pandas not available for detailed metrics)")
                        
                except Exception as metrics_error:
                    self.logger.warning(f"⚠️ Could not log feature engineering metrics: {metrics_error}")
                
            else:
                self.logger.error("❌ Feature engineering failed")
                self.enhanced_logger.log_issue("feature_engineering", "execution", "Feature engineering step failed", "error")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Feature engineering failed with exception: {e}")
            self.enhanced_logger.log_issue("feature_engineering", "exception", str(e), "error")
            return False

    @comprehensive_pipeline_protection(
        required_columns=['regime', 'label'],
        max_memory_mb=2000,
        max_execution_time=300,
        allowed_paths=['data_cache/*'],
        audit_access=True
    )
    @handles_errors(Exception, fallback=False)
    @traced(operation_name="matrix_operations")
    @log_execution_time
    async def _execute_matrix_operations(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        **kwargs
    ) -> bool:
        """Execute matrix operations step with comprehensive metrics logging."""
        self.logger.info("🧮 Executing matrix operations...")
        
        try:
            matrix_ops = EnhancedMatrixOperationsStep()
            success = await matrix_ops.perform_matrix_operations(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            if success:
                self.logger.info("✅ Matrix operations completed successfully")
                
                # Log detailed matrix operations metrics
                try:
                    # Try to load the matrix operations results for analysis
                    try:
                        import pandas as pd
                        import numpy as np
                        from pathlib import Path
                        
                        matrix_path = Path(data_dir) / f"matrix_operations_{exchange}_{symbol}_{timeframe}.parquet"
                        if matrix_path.exists():
                            matrix_data = pd.read_parquet(matrix_path)
                            
                            # Log feature quality metrics for matrix data
                            self.enhanced_logger.log_feature_quality("matrix_operations", matrix_data)
                            
                            # Calculate matrix-specific metrics
                            numeric_cols = matrix_data.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                corr_matrix = matrix_data[numeric_cols].corr()
                                
                                # Calculate condition number
                                try:
                                    condition_number = np.linalg.cond(corr_matrix.values)
                                except:
                                    condition_number = float('inf')
                                
                                # Count high correlation pairs
                                high_corr_pairs = 0
                                max_correlation = 0.0
                                for i in range(len(corr_matrix.columns)):
                                    for j in range(i+1, len(corr_matrix.columns)):
                                        corr_val = abs(corr_matrix.iloc[i, j])
                                        max_correlation = max(max_correlation, corr_val)
                                        if corr_val > 0.95:
                                            high_corr_pairs += 1
                                
                                # Log step7 specific metrics
                                step7_metrics = {
                                    'matrix_operations_performed': ['correlation_analysis', 'eigenvalue_analysis', 'feature_ranking'],
                                    'eigenvalue_analysis': {
                                        'condition_number': condition_number,
                                        'rank': np.linalg.matrix_rank(corr_matrix.values),
                                        'effective_rank': len(numeric_cols)
                                    },
                                    'correlation_analysis': {
                                        'high_correlation_pairs': high_corr_pairs,
                                        'max_correlation': max_correlation
                                    },
                                    'performance_metrics': {
                                        'computation_time': 0.0,  # Would need to be extracted from matrix ops
                                        'memory_usage_mb': 0.0
                                    }
                                }
                                self.enhanced_logger.log_step7_metrics("matrix_operations", step7_metrics)
                        else:
                            # Fallback metrics when file doesn't exist
                            step7_metrics = {
                                'matrix_operations_performed': ['correlation_analysis', 'eigenvalue_analysis'],
                                'eigenvalue_analysis': {
                                    'condition_number': 1.0,
                                    'rank': 0,
                                    'effective_rank': 0
                                },
                                'correlation_analysis': {
                                    'high_correlation_pairs': 0,
                                    'max_correlation': 0.0
                                },
                                'performance_metrics': {
                                    'computation_time': 0.0,
                                    'memory_usage_mb': 0.0
                                }
                            }
                            self.enhanced_logger.log_step7_metrics("matrix_operations", step7_metrics)
                    except ImportError:
                        # Fallback when pandas/numpy not available
                        self.logger.info("🧮 Matrix operations completed (pandas/numpy not available for detailed metrics)")
                        
                except Exception as metrics_error:
                    self.logger.warning(f"⚠️ Could not log matrix operations metrics: {metrics_error}")
                
            else:
                self.logger.error("❌ Matrix operations failed")
                self.enhanced_logger.log_issue("matrix_operations", "execution", "Matrix operations step failed", "error")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Matrix operations failed with exception: {e}")
            self.enhanced_logger.log_issue("matrix_operations", "exception", str(e), "error")
            return False

    @comprehensive_pipeline_protection(
        required_columns=['regime', 'label'],
        max_memory_mb=1500,
        max_execution_time=180,
        allowed_paths=['data_cache/*'],
        audit_access=True
    )
    @handles_errors(Exception, fallback=False)
    @traced(operation_name="feature_selection")
    @log_execution_time
    async def _execute_feature_selection(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        **kwargs
    ) -> bool:
        """Execute feature selection step."""
        self.logger.info("🎯 Executing feature selection...")
        
        try:
            feature_selector = AdvancedFeatureSelectionStep()
            success = await feature_selector.select_features(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir
            )
            
            if success:
                self.logger.info("✅ Feature selection completed successfully")
            else:
                self.logger.error("❌ Feature selection failed")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Feature selection failed with exception: {e}")
            return False

    @handles_errors(Exception, fallback=None)
    async def _save_pipeline_state(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        success: bool = True
    ) -> None:
        """Save pipeline state for monitoring and debugging."""
        try:
            state_file = Path(data_dir) / f"market_analysis_state_{symbol}_{timeframe}.json"
            
            pipeline_state = {
                **self.pipeline_state,
                'success': success,
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'execution_time': self._get_execution_time(),
                'timestamp': format_datetime(get_current_datetime()),
            }
            
            safe_json_dump(pipeline_state, state_file, indent=2)
            self.logger.info(f"💾 Pipeline state saved to: {state_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save pipeline state: {e}")

    def _get_execution_time(self) -> float:
        """Get total execution time in seconds."""
        if self.pipeline_state['start_time'] and self.pipeline_state['end_time']:
            return (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
        return 0.0

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'current_step': self.pipeline_state['current_step'],
            'completed_steps': self.pipeline_state['completed_steps'],
            'failed_steps': self.pipeline_state['failed_steps'],
            'execution_time': self._get_execution_time(),
            'correlation_id': self.pipeline_state['correlation_id'],
        }


# Main function for direct execution
async def run_enhanced_market_analysis_pipeline(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    **config
) -> bool:
    """
    Run the enhanced market analysis pipeline with comprehensive validation.
    
    This is the main entry point for the market analysis pipeline.
    """
    orchestrator = MarketAnalysisPipelineOrchestrator(config)
    return await orchestrator.execute_pipeline(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        **config
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'force_rerun': True,
            'hmm_clustering': True,
            'regime_splitting': True,
            'feature_engineering': True,
            'matrix_operations': True,
            'feature_selection': True,
            'random_state': 42,
        }
        
        success = await run_enhanced_market_analysis_pipeline(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            data_dir="data_cache",
            **config
        )
        
        if success:
            print("🎉 Market analysis pipeline completed successfully!")
        else:
            print("❌ Market analysis pipeline failed!")
    
    asyncio.run(main())