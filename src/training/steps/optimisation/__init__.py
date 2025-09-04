#!/usr/bin/env python3
"""Optimization Package for Trading Pipeline.

This package contains all the components for optimization:
- Confidence calibration per regime
- Final parameters optimization
- Parameter optimization wrapper
"""

from .step16_confidence_calibration_per_regime import ConfidenceCalibrationPerRegimeStep
from .step16_confidence_calibration_validator import ConfidenceCalibrationValidator
from .step17_final_parameters_optimization_new import FinalParametersOptimizationStep
from .step17_final_parameters_optimization_per_regime import PerRegimeFinalParametersOptimizationStep
from .step17_final_parameters_optimization_validator import FinalParametersOptimizationValidator
from .step17_parameter_optimization_wrapper import ParameterOptimizationWrapper
from .optimisation_pipeline_validator import OptimisationPipelineValidator
from .step_validators import (
    ConfidenceCalibrationStepValidator,
    FinalParametersOptimizationStepValidator,
    OptimisationPipelineStepValidator,
    create_optimisation_validator
)

# Enhanced main pipeline function with comprehensive validation and protection
async def run_optimisation_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete optimization pipeline with enhanced validation and protection."""
    from src.utils.common_operations import (
        format_datetime, get_current_datetime, safe_file_exists, 
        ensure_directory, safe_json_dump, safe_json_load, safe_sleep
    )
    from src.utils.data_quality_framework import DataQualityFramework
    from src.utils.logger import system_logger
    from src.core.decorators import handles_errors, validates, traced, log_execution_time
    
    logger = system_logger.getChild('OptimisationPipeline')
    
    @handles_errors(fallback=False, context="optimisation_pipeline_step")
    @traced(span_name="optimisation_pipeline_step")
    async def execute_optimisation_step(step_name: str, step_func, *args, **kwargs):
        """Execute a single optimisation step with comprehensive error handling."""
        logger.info(f"🚀 Starting {step_name}...")
        start_time = get_current_datetime()
        
        try:
            result = await step_func(*args, **kwargs)
            end_time = get_current_datetime()
            duration = (end_time - start_time).total_seconds()
            
            if result:
                logger.info(f"✅ {step_name} completed successfully in {duration:.2f}s")
                return True
            else:
                logger.error(f"❌ {step_name} failed")
                return False
                
        except Exception as e:
            end_time = get_current_datetime()
            duration = (end_time - start_time).total_seconds()
            logger.exception(f"❌ {step_name} failed with exception after {duration:.2f}s: {e}")
            return False
    
    @validates()
    async def validate_pipeline_data(symbol, exchange, data_dir):
        """Validate data quality before starting optimisation."""
        logger.info("🔍 Validating pipeline data quality...")
        
        dq_framework = DataQualityFramework()
        
        # Check data files
        data_files = [
            f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"{data_dir}/volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        for file_path in data_files:
            if not safe_file_exists(file_path):
                logger.error(f"❌ Required data file missing: {file_path}")
                return False
            
            # Basic data quality check
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                if df.empty:
                    logger.error(f"❌ Data file is empty: {file_path}")
                    return False
                
                # Check for critical data quality issues
                quality_result = dq_framework.validate_data(df, ['klines_schema'])
                if not quality_result.get('overall_passed', False):
                    logger.warning(f"⚠️ Data quality issues in {file_path}: {quality_result.get('errors', [])}")
                
            except Exception as e:
                logger.error(f"❌ Error reading data file {file_path}: {e}")
                return False
        
        logger.info("✅ Pipeline data validation passed")
        return True
    
    try:
        logger.info("🚀 Starting enhanced optimisation pipeline")
        logger.info(f"📊 Configuration: {config}")
        
        # Pre-pipeline validation
        if config.get('validation_enabled', True):
            validation_success = await validate_pipeline_data(symbol, exchange, data_dir)
            if not validation_success:
                logger.error("❌ Pre-pipeline validation failed")
                return False
        
        # Initialize data quality framework for monitoring
        dq_framework = DataQualityFramework()
        
        # Step 1: Confidence Calibration (if enabled)
        if config.get('confidence_calibration', True):
            logger.info("🎯 Step 1: Confidence Calibration")
            
            # Create enhanced confidence calibrator with data protection
            confidence_calibrator = ConfidenceCalibrationPerRegimeStep(config)
            
            # Execute with comprehensive error handling
            step1_success = await execute_optimisation_step(
                "Confidence Calibration",
                confidence_calibrator.calibrate_confidence,
                symbol, exchange, timeframe, data_dir
            )
            
            if not step1_success:
                logger.error("❌ Confidence calibration failed")
                if config.get('strict_mode', False):
                    return False
                else:
                    logger.warning("⚠️ Continuing with default confidence parameters")
        
        # Step 2: Final Parameters Optimization (if enabled)
        if config.get('parameter_optimization', True):
            logger.info("🎯 Step 2: Final Parameters Optimization")
            
            # Create enhanced parameter optimizer with data protection
            param_optimizer = FinalParametersOptimizationStep(config)
            
            # Execute with comprehensive error handling
            step2_success = await execute_optimisation_step(
                "Final Parameters Optimization",
                param_optimizer.optimize_parameters,
                symbol, exchange, timeframe, data_dir
            )
            
            if not step2_success:
                logger.error("❌ Final parameters optimization failed")
                if config.get('strict_mode', False):
                    return False
                else:
                    logger.warning("⚠️ Continuing with default parameters")
        
        # Post-pipeline validation and cleanup
        if config.get('post_validation', True):
            logger.info("🔍 Performing post-pipeline validation...")
            
            # Check that output files were created
            expected_outputs = [
                f"models/{symbol}_{exchange}_confidence_calibration.json",
                f"models/{symbol}_{exchange}_optimized_parameters.json"
            ]
            
            missing_outputs = []
            for output_file in expected_outputs:
                if not safe_file_exists(output_file):
                    missing_outputs.append(output_file)
            
            if missing_outputs:
                logger.warning(f"⚠️ Some output files missing: {missing_outputs}")
            else:
                logger.info("✅ All expected output files created")
        
        # Save pipeline execution summary
        if config.get('save_summary', True):
            summary_file = f"{data_dir}/optimisation_pipeline_summary_{symbol}_{timeframe}.json"
            summary_data = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'config': config,
                'execution_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                'success': True,
                'steps_completed': {
                    'confidence_calibration': config.get('confidence_calibration', True),
                    'parameter_optimization': config.get('parameter_optimization', True)
                }
            }
            
            safe_json_dump(summary_data, summary_file, indent=2)
            logger.info(f"💾 Pipeline summary saved to: {summary_file}")
        
        logger.info("🎉 Enhanced optimisation pipeline completed successfully")
        return True
        
    except Exception as e:
        logger.exception(f"❌ Enhanced optimisation pipeline failed: {e}")
        return False

__all__ = [
    'ConfidenceCalibrationPerRegimeStep',
    'ConfidenceCalibrationValidator',
    'FinalParametersOptimizationStep',
    'PerRegimeFinalParametersOptimizationStep',
    'FinalParametersOptimizationValidator',
    'ParameterOptimizationWrapper',
    'OptimisationPipelineValidator',
    'ConfidenceCalibrationStepValidator',
    'FinalParametersOptimizationStepValidator',
    'OptimisationPipelineStepValidator',
    'create_optimisation_validator',
    'run_optimisation_pipeline'
]