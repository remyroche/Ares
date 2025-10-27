"""
Unified Training Pipeline - Main Entry Point

This module provides the main entry point for the unified training pipeline,
replacing the fragmented training components with a single, comprehensive
orchestration system.

Key Features:
- Single entry point for all training operations
- Unified configuration management
- Comprehensive error handling and monitoring
- Role-specific training coordination
- Performance optimization and resource management
- Advanced mathematical validation and safety
- Comprehensive data processing and quality validation
- Hardware-optimized computations with VectorBT integration
- Hyperparameter optimization with HPO tools
- ML utilities for explainability, time series, and validation
"""

import asyncio
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime
import time

# Core utilities
from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_performance, tprint_data_preview, tprint_data_format,
    tprint_progress, tprint_exception, tprint_structured, tprint_timer
)
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int,
    get_memory_usage, optimize_dataframe_memory, memory_checkpoint
)
from src.utils.common_utilities import calculate_data_quality_metrics, get_dataframe_info

# Mathematical validation and safety
from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range, validate_array_finite,
    validate_matrix_finite, validate_probability, validate_numeric_array,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    safe_correlation, safe_covariance, safe_percentile, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, check_for_inf_nan,
    check_for_nans, check_for_infs, is_valid_number, MathValidation
)

# Data processing and quality validation
from src.utils.data.unified_data_utils import UnifiedDataUtils
from src.utils.data.quality.data_quality import DataQualityFramework, QualityResult, QualityThresholds
from src.utils.data.processing.data_processing import DataProcessor
from src.utils.data.quality.data_cleaning import DataCleaner

# Hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, get_unified_hardware_manager,
    WorkloadType, OptimizationLevel, HardwareConfig
)
from src.utils.hardware.m1_memory_optimizer import optimize_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager

# ML common utilities and optimization
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, get_unified_vectorization_manager,
    OperationType, OptimizationStrategy, StrategySelectionConfig
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.ml_common.optimization.auto_tuner import AutoTuner
from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO
from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPrevention

# ML utilities for explainability and validation
from src.utils.ml_common.explainability.shap_lime_integration import SHAPLIMEIntegration
from src.utils.ml_common.explainability.model_explainability import ModelExplainability
from src.utils.ml_common.validation.purged_kfold import PurgedKFold
from src.utils.ml_common.validation.lookahead_bias_detector import LookaheadBiasDetector
from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector

# Time series and ensemble utilities
from src.utils.ml_common.ensembles.enhanced_oof_stacking_with_confidence import EnhancedOOFStacking
from src.utils.ml_common.ensembles.vectorbt_ensemble_optimizer import VectorBTEnsembleOptimizer
from src.utils.ml_common.evaluation.unified_evaluator import UnifiedEvaluator

# Data access and storage
from src.utils.kline_parquet import KlinesParquetManager
from src.core.decorators import handles_errors, traced, log_execution_time

from .core import (
    TrainingPipelineOrchestrator, PipelineConfig, PipelineResult,
    PipelinePhase, PipelineStatus, TrainingRole, ModelType
)


class UnifiedTrainingPipeline:
    """
    Unified training pipeline main entry point.
    
    This class provides a single, comprehensive interface for all training
    operations, replacing the fragmented training components with a unified
    orchestration system.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the unified training pipeline with comprehensive utility integration.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or system_logger.getChild("UnifiedTrainingPipeline")
        self._orchestrator = None
        
        # Initialize utility managers
        self.hardware_manager = get_unified_hardware_manager()
        self.vectorization_manager = get_unified_vectorization_manager()
        self.data_utils = UnifiedDataUtils()
        self.math_validator = MathValidation()
        
        # Legacy hardware optimizers (for backward compatibility)
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.gpu_manager = get_m1_gpu_manager()
        
        # ML optimization tools
        self.bayesian_optimizer = BayesianTPEOptimizer()
        self.auto_tuner = AutoTuner()
        self.hierarchical_hpo = HierarchicalHPO()
        self.overfitting_prevention = OverfittingPrevention()
        
        # ML utilities
        self.shap_lime = SHAPLIMEIntegration()
        self.model_explainability = ModelExplainability()
        self.purged_kfold = PurgedKFold()
        self.lookahead_detector = LookaheadBiasDetector()
        self.data_leakage_detector = DataLeakageDetector()
        
        # Ensemble and evaluation
        self.oof_stacking = EnhancedOOFStacking()
        self.vectorbt_ensemble = VectorBTEnsembleOptimizer()
        self.unified_evaluator = UnifiedEvaluator()
        
        # Data access
        self.klines_manager = KlinesParquetManager()
        
        self._validate_initialization()
        self.logger.info("Initialized UnifiedTrainingPipeline with comprehensive utilities")
    
    def _validate_initialization(self) -> None:
        """Validate that all utility managers are properly initialized."""
        try:
            # Validate core managers
            assert self.hardware_manager is not None, "Hardware manager not initialized"
            assert self.vectorization_manager is not None, "Vectorization manager not initialized"
            assert self.data_utils is not None, "Data utils not initialized"
            assert self.math_validator is not None, "Math validator not initialized"
            
            # Validate ML tools
            assert self.bayesian_optimizer is not None, "Bayesian optimizer not initialized"
            assert self.auto_tuner is not None, "Auto tuner not initialized"
            assert self.hierarchical_hpo is not None, "Hierarchical HPO not initialized"
            
            # Validate ML utilities
            assert self.shap_lime is not None, "SHAP/LIME integration not initialized"
            assert self.model_explainability is not None, "Model explainability not initialized"
            assert self.purged_kfold is not None, "Purged KFold not initialized"
            assert self.lookahead_detector is not None, "Lookahead detector not initialized"
            assert self.data_leakage_detector is not None, "Data leakage detector not initialized"
            
            tprint_success("All utility managers validated successfully")
            
        except Exception as e:
            tprint_error(f"Initialization validation failed: {e}")
            raise RuntimeError(f"Failed to validate utility managers: {e}")
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=PipelineResult(
            success=False,
            status=PipelineStatus.FAILED,
            execution_time=0.0,
            phases_completed=[],
            phases_failed=[PipelinePhase.INITIALIZATION],
            errors=["Pipeline initialization failed"]
        ),
        context="unified training pipeline"
    )
    async def execute_training_pipeline(
        self,
        data: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        analyst_targets: Optional[pd.Series] = None,
        tactician_targets: Optional[pd.Series] = None
    ) -> PipelineResult:
        """
        Execute the complete training pipeline with comprehensive utility integration.
        
        Args:
            data: Training data
            config: Pipeline configuration (optional)
            analyst_targets: Analyst target variables (optional)
            tactician_targets: Tactician target variables (optional)
            
        Returns:
            Pipeline execution result
        """
        start_time = time.time()
        
        try:
            tprint_info("🚀 Starting unified training pipeline with comprehensive utilities...")
            
            # Hardware optimization context
            with self.hardware_manager.optimization_context(
                workload_type=WorkloadType.ML_TRAINING,
                optimization_level=OptimizationLevel.AGGRESSIVE
            ):
                # Data validation and preprocessing
                tprint_info("📊 Validating and preprocessing data...")
                validated_data = await self._validate_and_preprocess_data(data)
                
                # Data quality assessment
                tprint_info("🔍 Assessing data quality...")
                quality_result = await self._assess_data_quality(validated_data)
                
                # Mathematical validation
                tprint_info("🧮 Performing mathematical validation...")
                math_validated_data = await self._validate_mathematical_operations(validated_data)
                
                # Create pipeline configuration with utility settings
                pipeline_config = self._create_enhanced_pipeline_config(config, quality_result)
                
                # Create orchestrator
                self._orchestrator = TrainingPipelineOrchestrator(pipeline_config, self.logger)
                
                # Execute pipeline with enhanced monitoring
                result = await self._execute_enhanced_pipeline(
                    math_validated_data, analyst_targets, tactician_targets, quality_result
                )
                
                # Post-processing and evaluation
                if result.success:
                    tprint_info("📈 Performing post-training analysis...")
                    await self._post_training_analysis(result, math_validated_data)
                
                execution_time = time.time() - start_time
                result.execution_time = execution_time
                
                if result.success:
                    tprint_success(f"✅ Training pipeline completed successfully in {execution_time:.2f}s")
                else:
                    tprint_error(f"❌ Training pipeline failed after {execution_time:.2f}s")
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_exception(f"Unified training pipeline failed after {execution_time:.2f}s: {e}")
            return PipelineResult(
                success=False,
                status=PipelineStatus.FAILED,
                execution_time=execution_time,
                phases_completed=[],
                phases_failed=[PipelinePhase.INITIALIZATION],
                errors=[f"Pipeline execution failed: {e}"]
            )
    
    async def _validate_and_preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and preprocess data using comprehensive data utilities."""
        try:
            tprint_info("🔧 Validating and preprocessing data...")
            
            # Use unified data utils for comprehensive processing
            processed_data = await self.data_utils.process_and_validate(
                data=data,
                validation_config={
                    'check_missing': True,
                    'check_outliers': True,
                    'check_data_types': True,
                    'optimize_memory': True
                }
            )
            
            # Data format compatibility check
            tprint_data_format(processed_data, "Processed training data")
            
            return processed_data
            
        except Exception as e:
            tprint_error(f"Data validation and preprocessing failed: {e}")
            raise
    
    async def _assess_data_quality(self, data: pd.DataFrame) -> QualityResult:
        """Assess data quality using comprehensive quality framework."""
        try:
            tprint_info("📊 Assessing data quality...")
            
            # Use data quality framework
            quality_result = await self.data_utils.validate_data_quality(
                data=data,
                thresholds=QualityThresholds(
                    min_completeness=0.95,
                    max_outlier_ratio=0.05,
                    min_correlation_strength=0.1
                )
            )
            
            # Log quality metrics
            tprint_structured({
                "completeness": quality_result.completeness_score,
                "outlier_ratio": quality_result.outlier_ratio,
                "correlation_strength": quality_result.correlation_strength,
                "overall_quality": quality_result.overall_quality
            }, "Data Quality Assessment")
            
            return quality_result
            
        except Exception as e:
            tprint_error(f"Data quality assessment failed: {e}")
            raise
    
    async def _validate_mathematical_operations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate mathematical operations and ensure numerical stability."""
        try:
            tprint_info("🧮 Validating mathematical operations...")
            
            # Validate all numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                # Check for finite values
                if not validate_array_finite(data[col].values):
                    tprint_warning(f"Non-finite values found in column {col}, cleaning...")
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                    data[col] = data[col].fillna(data[col].median())
                
                # Validate range if applicable
                if col in ['price', 'volume', 'amount']:
                    validate_positive(data[col].values, f"Column {col} should be positive")
            
            # Data preview after mathematical validation
            tprint_data_preview(data, "Mathematically validated data")
            
            return data
            
        except Exception as e:
            tprint_error(f"Mathematical validation failed: {e}")
            raise
    
    def _create_enhanced_pipeline_config(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        quality_result: Optional[QualityResult] = None
    ) -> PipelineConfig:
        """Create enhanced pipeline configuration with utility settings."""
        try:
            # Default configuration
            default_config = {
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'execution_mode': 'full',
                'enable_analyst': True,
                'enable_tactician': True,
                'enable_ensemble': True,
                'max_parallel_tasks': 3,
                'enable_monitoring': True,
                'monitoring_interval': 30.0,
                'enable_health_checks': True,
                'enable_vectorization': True,
                'enable_hpo': True,
                'enable_explainability': True,
                'enable_data_leakage_detection': True,
                'enable_lookahead_detection': True
            }
            
            # Merge with provided configuration
            if config:
                default_config.update(config)
            
            # Adjust based on data quality
            if quality_result and quality_result.overall_quality < 0.8:
                tprint_warning("Data quality below threshold, enabling additional validation")
                default_config['enable_data_leakage_detection'] = True
                default_config['enable_lookahead_detection'] = True
            
            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                symbol=default_config.get('symbol', 'ETHUSDT'),
                timeframe=default_config.get('timeframe', '15m'),
                execution_mode=default_config.get('execution_mode', 'full'),
                enable_analyst=default_config.get('enable_analyst', True),
                enable_tactician=default_config.get('enable_tactician', True),
                enable_ensemble=default_config.get('enable_ensemble', True),
                max_parallel_tasks=default_config.get('max_parallel_tasks', 3),
                memory_limit_mb=default_config.get('memory_limit_mb'),
                timeout_seconds=default_config.get('timeout_seconds'),
                enable_monitoring=default_config.get('enable_monitoring', True),
                monitoring_interval=default_config.get('monitoring_interval', 30.0),
                enable_health_checks=default_config.get('enable_health_checks', True)
            )
            
            # Add utility-specific settings
            pipeline_config.utility_settings = {
                'enable_vectorization': default_config.get('enable_vectorization', True),
                'enable_hpo': default_config.get('enable_hpo', True),
                'enable_explainability': default_config.get('enable_explainability', True),
                'enable_data_leakage_detection': default_config.get('enable_data_leakage_detection', True),
                'enable_lookahead_detection': default_config.get('enable_lookahead_detection', True),
                'quality_result': quality_result
            }
            
            return pipeline_config
            
        except Exception as e:
            tprint_error(f"Failed to create enhanced pipeline config: {e}")
            raise
    
    async def _execute_enhanced_pipeline(
        self,
        data: pd.DataFrame,
        analyst_targets: Optional[pd.Series] = None,
        tactician_targets: Optional[pd.Series] = None,
        quality_result: Optional[QualityResult] = None
    ) -> PipelineResult:
        """Execute pipeline with enhanced monitoring and utilities."""
        try:
            # Check for data leakage and lookahead bias
            if self.data_leakage_detector:
                tprint_info("🔍 Checking for data leakage...")
                leakage_result = await self.data_leakage_detector.detect_leakage(data)
                if leakage_result.has_leakage:
                    tprint_warning(f"Data leakage detected: {leakage_result.issues}")
            
            if self.lookahead_detector:
                tprint_info("🔍 Checking for lookahead bias...")
                lookahead_result = await self.lookahead_detector.detect_bias(data)
                if lookahead_result.has_bias:
                    tprint_warning(f"Lookahead bias detected: {lookahead_result.issues}")
            
            # Execute pipeline with vectorization optimization
            with self.vectorization_manager.optimization_context(
                operation_type=OperationType.ML_TRAINING,
                data_size=data.shape[0]
            ):
                result = await self._orchestrator.execute_pipeline(
                    data, analyst_targets, tactician_targets
                )
            
            return result
            
        except Exception as e:
            tprint_error(f"Enhanced pipeline execution failed: {e}")
            raise
    
    async def _post_training_analysis(
        self, 
        result: PipelineResult, 
        data: pd.DataFrame
    ) -> None:
        """Perform post-training analysis and explainability."""
        try:
            tprint_info("📈 Performing post-training analysis...")
            
            # Model explainability analysis
            if hasattr(result, 'models') and result.models:
                tprint_info("🔍 Generating model explanations...")
                for model_name, model in result.models.items():
                    try:
                        explanations = await self.model_explainability.generate_explanations(
                            model=model,
                            data=data,
                            method='shap'
                        )
                        tprint_structured(explanations, f"Model explanations for {model_name}")
                    except Exception as e:
                        tprint_warning(f"Failed to generate explanations for {model_name}: {e}")
            
            # Performance analysis
            if hasattr(result, 'performance_metrics'):
                tprint_structured(result.performance_metrics, "Training Performance Metrics")
            
            tprint_success("Post-training analysis completed")
            
        except Exception as e:
            tprint_warning(f"Post-training analysis failed: {e}")
            # Don't raise - this is not critical for pipeline success
    
    def _create_pipeline_config(self, config: Optional[Dict[str, Any]] = None) -> PipelineConfig:
        """Create pipeline configuration from input."""
        try:
            # Default configuration
            default_config = {
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'execution_mode': 'full',
                'enable_analyst': True,
                'enable_tactician': True,
                'enable_ensemble': True,
                'max_parallel_tasks': 3,
                'enable_monitoring': True,
                'monitoring_interval': 30.0,
                'enable_health_checks': True
            }
            
            # Merge with provided configuration
            if config:
                default_config.update(config)
            
            # Create pipeline configuration
            pipeline_config = PipelineConfig(
                symbol=default_config.get('symbol', 'ETHUSDT'),
                timeframe=default_config.get('timeframe', '15m'),
                execution_mode=default_config.get('execution_mode', 'full'),
                enable_analyst=default_config.get('enable_analyst', True),
                enable_tactician=default_config.get('enable_tactician', True),
                enable_ensemble=default_config.get('enable_ensemble', True),
                max_parallel_tasks=default_config.get('max_parallel_tasks', 3),
                memory_limit_mb=default_config.get('memory_limit_mb'),
                timeout_seconds=default_config.get('timeout_seconds'),
                enable_monitoring=default_config.get('enable_monitoring', True),
                monitoring_interval=default_config.get('monitoring_interval', 30.0),
                enable_health_checks=default_config.get('enable_health_checks', True),
                analyst_config=default_config.get('analyst_config'),
                tactician_config=default_config.get('tactician_config'),
                ensemble_config=default_config.get('ensemble_config'),
                custom_params=default_config.get('custom_params', {})
            )
            
            self.logger.info(f"Created pipeline configuration: {pipeline_config.symbol} {pipeline_config.timeframe}")
            return pipeline_config
            
        except Exception as e:
            self.logger.error(f"Pipeline configuration creation failed: {e}")
            # Return minimal configuration
            return PipelineConfig()
    
    async def train_analyst_models(
        self,
        data: pd.DataFrame,
        targets: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train analyst models with comprehensive utility integration.
        
        Args:
            data: Training data
            targets: Target variables
            config: Training configuration (optional)
            
        Returns:
            Training result with enhanced metrics and explanations
        """
        try:
            tprint_info("📊 Starting analyst models training with enhanced utilities...")
            
            # Validate targets mathematically
            if not validate_array_finite(targets.values):
                tprint_warning("Non-finite values in targets, cleaning...")
                targets = targets.replace([np.inf, -np.inf], np.nan).fillna(targets.median())
            
            # Data quality assessment for targets
            target_quality = await self.data_utils.validate_data_quality(
                data=pd.DataFrame({'target': targets}),
                thresholds=QualityThresholds(min_completeness=0.9)
            )
            
            if target_quality.overall_quality < 0.8:
                tprint_warning("Target quality below threshold, applying corrections...")
                targets = await self.data_utils.clean_data(pd.DataFrame({'target': targets}))['target']
            
            # Create analyst-only configuration with HPO
            analyst_config = {
                'enable_analyst': True,
                'enable_tactician': False,
                'enable_ensemble': False,
                'enable_hpo': True,
                'enable_explainability': True,
                'enable_vectorization': True,
                **(config or {})
            }
            
            # Execute pipeline with enhanced monitoring
            with tprint_timer("Analyst Models Training"):
                result = await self.execute_training_pipeline(data, analyst_config, targets)
            
            if result.success and result.analyst_result:
                # Generate model explanations
                if hasattr(result, 'analyst_models') and result.analyst_models:
                    tprint_info("🔍 Generating analyst model explanations...")
                    for model_name, model in result.analyst_models.items():
                        try:
                            explanations = await self.model_explainability.generate_explanations(
                                model=model,
                                data=data,
                                method='shap'
                            )
                            result.analyst_result[f'{model_name}_explanations'] = explanations
                        except Exception as e:
                            tprint_warning(f"Failed to generate explanations for {model_name}: {e}")
                
                # Add quality metrics
                result.analyst_result['target_quality'] = target_quality.overall_quality
                result.analyst_result['data_quality'] = result.analyst_result.get('data_quality', {})
                
                tprint_success("✅ Analyst models training completed with explanations")
                return result.analyst_result
            else:
                tprint_error("❌ Analyst models training failed")
                return {'success': False, 'error_message': 'Analyst training failed'}
                
        except Exception as e:
            tprint_exception(f"Analyst models training failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def train_tactician_models(
        self,
        data: pd.DataFrame,
        targets: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train tactician models with comprehensive utility integration.
        
        Args:
            data: Training data
            targets: Target variables
            config: Training configuration (optional)
            
        Returns:
            Training result with enhanced metrics and explanations
        """
        try:
            tprint_info("⚔️ Starting tactician models training with enhanced utilities...")
            
            # Validate targets mathematically
            if not validate_array_finite(targets.values):
                tprint_warning("Non-finite values in tactician targets, cleaning...")
                targets = targets.replace([np.inf, -np.inf], np.nan).fillna(targets.median())
            
            # Data quality assessment for targets
            target_quality = await self.data_utils.validate_data_quality(
                data=pd.DataFrame({'target': targets}),
                thresholds=QualityThresholds(min_completeness=0.9)
            )
            
            if target_quality.overall_quality < 0.8:
                tprint_warning("Tactician target quality below threshold, applying corrections...")
                targets = await self.data_utils.clean_data(pd.DataFrame({'target': targets}))['target']
            
            # Create tactician-only configuration with HPO and vectorization
            tactician_config = {
                'enable_analyst': False,
                'enable_tactician': True,
                'enable_ensemble': False,
                'enable_hpo': True,
                'enable_explainability': True,
                'enable_vectorization': True,
                'enable_data_leakage_detection': True,
                'enable_lookahead_detection': True,
                **(config or {})
            }
            
            # Execute pipeline with enhanced monitoring
            with tprint_timer("Tactician Models Training"):
                result = await self.execute_training_pipeline(data, tactician_config, tactician_targets=targets)
            
            if result.success and result.tactician_result:
                # Generate model explanations
                if hasattr(result, 'tactician_models') and result.tactician_models:
                    tprint_info("🔍 Generating tactician model explanations...")
                    for model_name, model in result.tactician_models.items():
                        try:
                            explanations = await self.model_explainability.generate_explanations(
                                model=model,
                                data=data,
                                method='shap'
                            )
                            result.tactician_result[f'{model_name}_explanations'] = explanations
                        except Exception as e:
                            tprint_warning(f"Failed to generate explanations for {model_name}: {e}")
                
                # Add quality metrics
                result.tactician_result['target_quality'] = target_quality.overall_quality
                result.tactician_result['data_quality'] = result.tactician_result.get('data_quality', {})
                
                tprint_success("✅ Tactician models training completed with explanations")
                return result.tactician_result
            else:
                tprint_error("❌ Tactician models training failed")
                return {'success': False, 'error_message': 'Tactician training failed'}
                
        except Exception as e:
            tprint_exception(f"Tactician models training failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def train_ensemble_models(
        self,
        data: pd.DataFrame,
        analyst_targets: pd.Series,
        tactician_targets: Optional[pd.Series] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble models with comprehensive utility integration.
        
        Args:
            data: Training data
            analyst_targets: Analyst target variables
            tactician_targets: Tactician target variables (optional)
            config: Training configuration (optional)
            
        Returns:
            Training result with enhanced metrics and explanations
        """
        try:
            tprint_info("🎯 Starting ensemble models training with enhanced utilities...")
            
            # Validate all targets mathematically
            if not validate_array_finite(analyst_targets.values):
                tprint_warning("Non-finite values in analyst targets, cleaning...")
                analyst_targets = analyst_targets.replace([np.inf, -np.inf], np.nan).fillna(analyst_targets.median())
            
            if tactician_targets is not None and not validate_array_finite(tactician_targets.values):
                tprint_warning("Non-finite values in tactician targets, cleaning...")
                tactician_targets = tactician_targets.replace([np.inf, -np.inf], np.nan).fillna(tactician_targets.median())
            
            # Data quality assessment for all targets
            target_quality = await self.data_utils.validate_data_quality(
                data=pd.DataFrame({'analyst_target': analyst_targets}),
                thresholds=QualityThresholds(min_completeness=0.9)
            )
            
            if tactician_targets is not None:
                tactician_quality = await self.data_utils.validate_data_quality(
                    data=pd.DataFrame({'tactician_target': tactician_targets}),
                    thresholds=QualityThresholds(min_completeness=0.9)
                )
                target_quality.overall_quality = min(target_quality.overall_quality, tactician_quality.overall_quality)
            
            if target_quality.overall_quality < 0.8:
                tprint_warning("Target quality below threshold, applying corrections...")
                analyst_targets = await self.data_utils.clean_data(pd.DataFrame({'target': analyst_targets}))['target']
                if tactician_targets is not None:
                    tactician_targets = await self.data_utils.clean_data(pd.DataFrame({'target': tactician_targets}))['target']
            
            # Create ensemble-only configuration with advanced features
            ensemble_config = {
                'enable_analyst': False,
                'enable_tactician': False,
                'enable_ensemble': True,
                'enable_hpo': True,
                'enable_explainability': True,
                'enable_vectorization': True,
                'enable_oof_stacking': True,
                'enable_vectorbt_optimization': True,
                'enable_data_leakage_detection': True,
                'enable_lookahead_detection': True,
                **(config or {})
            }
            
            # Execute pipeline with enhanced monitoring
            with tprint_timer("Ensemble Models Training"):
                result = await self.execute_training_pipeline(
                    data, ensemble_config, analyst_targets, tactician_targets
                )
            
            if result.success and result.ensemble_result:
                # Generate ensemble model explanations
                if hasattr(result, 'ensemble_models') and result.ensemble_models:
                    tprint_info("🔍 Generating ensemble model explanations...")
                    for model_name, model in result.ensemble_models.items():
                        try:
                            explanations = await self.model_explainability.generate_explanations(
                                model=model,
                                data=data,
                                method='shap'
                            )
                            result.ensemble_result[f'{model_name}_explanations'] = explanations
                        except Exception as e:
                            tprint_warning(f"Failed to generate explanations for {model_name}: {e}")
                
                # Add quality metrics and ensemble-specific metrics
                result.ensemble_result['target_quality'] = target_quality.overall_quality
                result.ensemble_result['data_quality'] = result.ensemble_result.get('data_quality', {})
                
                # Add ensemble performance metrics
                if hasattr(result, 'ensemble_performance'):
                    result.ensemble_result['ensemble_performance'] = result.ensemble_performance
                
                tprint_success("✅ Ensemble models training completed with explanations")
                return result.ensemble_result
            else:
                tprint_error("❌ Ensemble models training failed")
                return {'success': False, 'error_message': 'Ensemble training failed'}
                
        except Exception as e:
            tprint_exception(f"Ensemble models training failed: {e}")
            return {'success': False, 'error_message': str(e)}
    
    async def train_all_models(
        self,
        data: pd.DataFrame,
        analyst_targets: pd.Series,
        tactician_targets: Optional[pd.Series] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Train all models (analyst, tactician, ensemble) with comprehensive utility integration.
        
        Args:
            data: Training data
            analyst_targets: Analyst target variables
            tactician_targets: Tactician target variables (optional)
            config: Training configuration (optional)
            
        Returns:
            Complete pipeline result with enhanced metrics and explanations
        """
        try:
            tprint_info("🚀 Starting complete training pipeline with comprehensive utilities...")
            
            # Enhanced configuration for full pipeline
            enhanced_config = {
                'enable_analyst': True,
                'enable_tactician': True,
                'enable_ensemble': True,
                'enable_hpo': True,
                'enable_explainability': True,
                'enable_vectorization': True,
                'enable_oof_stacking': True,
                'enable_vectorbt_optimization': True,
                'enable_data_leakage_detection': True,
                'enable_lookahead_detection': True,
                'enable_hardware_optimization': True,
                'enable_math_validation': True,
                'enable_data_quality_validation': True,
                **(config or {})
            }
            
            # Execute full pipeline with comprehensive monitoring
            with tprint_timer("Complete Training Pipeline"):
                result = await self.execute_training_pipeline(
                    data, enhanced_config, analyst_targets, tactician_targets
                )
            
            if result.success:
                # Generate comprehensive analysis
                tprint_info("📊 Generating comprehensive training analysis...")
                await self._generate_comprehensive_analysis(result, data)
                
                tprint_success("✅ Complete training pipeline completed with comprehensive analysis")
            else:
                tprint_error("❌ Complete training pipeline failed")
            
            return result
            
        except Exception as e:
            tprint_exception(f"Complete training pipeline failed: {e}")
            return PipelineResult(
                success=False,
                status=PipelineStatus.FAILED,
                execution_time=0.0,
                phases_completed=[],
                phases_failed=[PipelinePhase.INITIALIZATION],
                errors=[f"Complete training pipeline failed: {e}"]
            )
    
    async def _generate_comprehensive_analysis(
        self, 
        result: PipelineResult, 
        data: pd.DataFrame
    ) -> None:
        """Generate comprehensive analysis of training results."""
        try:
            tprint_info("📊 Generating comprehensive training analysis...")
            
            # Performance summary
            performance_summary = {
                'execution_time': result.execution_time,
                'phases_completed': len(result.phases_completed),
                'phases_failed': len(result.phases_failed),
                'success_rate': len(result.phases_completed) / (len(result.phases_completed) + len(result.phases_failed)) if (result.phases_completed or result.phases_failed) else 0
            }
            
            tprint_structured(performance_summary, "Training Performance Summary")
            
            # Model performance analysis
            if hasattr(result, 'analyst_result') and result.analyst_result:
                tprint_structured(result.analyst_result, "Analyst Models Performance")
            
            if hasattr(result, 'tactician_result') and result.tactician_result:
                tprint_structured(result.tactician_result, "Tactician Models Performance")
            
            if hasattr(result, 'ensemble_result') and result.ensemble_result:
                tprint_structured(result.ensemble_result, "Ensemble Models Performance")
            
            # Data quality summary
            if hasattr(result, 'data_quality_summary'):
                tprint_structured(result.data_quality_summary, "Data Quality Summary")
            
            # Hardware optimization summary
            if hasattr(result, 'hardware_optimization_summary'):
                tprint_structured(result.hardware_optimization_summary, "Hardware Optimization Summary")
            
            tprint_success("Comprehensive analysis completed")
            
        except Exception as e:
            tprint_warning(f"Comprehensive analysis failed: {e}")
            # Don't raise - this is not critical for pipeline success
    
    def get_pipeline_status(self) -> Optional[Dict[str, Any]]:
        """Get current pipeline status."""
        if self._orchestrator:
            return self._orchestrator.get_pipeline_status()
        return None
    
    def get_required_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return [
            'pandas', 'numpy', 'scikit-learn', 'lightgbm', 'catboost',
            'torch', 'psutil', 'asyncio'
        ]
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get pipeline processing capabilities."""
        return {
            'supports_parallel_processing': True,
            'supports_monitoring': True,
            'supports_health_checks': True,
            'supports_ensemble': True,
            'memory_efficient': True,
            'role_specific_training': True
        }


# Convenience functions for easy usage
async def create_unified_training_pipeline(
    logger: Optional[logging.Logger] = None
) -> UnifiedTrainingPipeline:
    """Create a new unified training pipeline instance."""
    return UnifiedTrainingPipeline(logger)


async def execute_quick_training(
    data: pd.DataFrame,
    analyst_targets: pd.Series,
    tactician_targets: Optional[pd.Series] = None,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    logger: Optional[logging.Logger] = None,
    enable_artifact_chaining: bool = False
) -> PipelineResult:
    """
    Execute quick training with minimal configuration.
    
    Args:
        data: Training data
        analyst_targets: Analyst target variables
        tactician_targets: Tactician target variables (optional)
        symbol: Trading symbol
        timeframe: Trading timeframe
        logger: Logger instance (optional)
        
    Returns:
        Pipeline execution result
    """
    pipeline = UnifiedTrainingPipeline(logger)
    
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'execution_mode': 'light',  # Quick training mode
        'enable_artifact_chaining': enable_artifact_chaining
    }
    
    return await pipeline.execute_training_pipeline(
        data, config, analyst_targets, tactician_targets
    )


async def execute_full_training(
    data: pd.DataFrame,
    analyst_targets: pd.Series,
    tactician_targets: Optional[pd.Series] = None,
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    logger: Optional[logging.Logger] = None,
    enable_artifact_chaining: bool = True
) -> PipelineResult:
    """
    Execute full training with comprehensive configuration.
    
    Args:
        data: Training data
        analyst_targets: Analyst target variables
        tactician_targets: Tactician target variables (optional)
        symbol: Trading symbol
        timeframe: Trading timeframe
        logger: Logger instance (optional)
        
    Returns:
        Pipeline execution result
    """
    pipeline = UnifiedTrainingPipeline(logger)
    
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'execution_mode': 'full',  # Full training mode
        'enable_monitoring': True,
        'enable_health_checks': True,
        'max_parallel_tasks': 3,
        'enable_artifact_chaining': enable_artifact_chaining
    }
    
    return await pipeline.execute_training_pipeline(
        data, config, analyst_targets, tactician_targets
    )
