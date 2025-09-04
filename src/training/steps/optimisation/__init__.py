#!/usr/bin/env python3
"""Optimization Package for Trading Pipeline.

This package contains all the components for optimization:
- Confidence calibration per regime
- Final parameters optimization
- Parameter optimization wrapper

Enhanced with comprehensive logging, progress tracking, and quality monitoring.
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
    """Run the complete optimization pipeline with enhanced validation and protection.
    
    Features:
    - 🎯 Comprehensive logging with emojis for easy troubleshooting
    - 📊 Real-time progress tracking and monitoring
    - 🔍 Detailed error reporting and quality issue flagging
    - ✅ Step-by-step validation with detailed reporting
    - 📈 Performance metrics tracking
    - 🛡️ Data quality monitoring throughout the process
    """
    from src.utils.common_operations import (
        format_datetime, get_current_datetime, safe_file_exists, 
        ensure_directory, safe_json_dump, safe_json_load, safe_sleep
    )
    from src.utils.data_quality_framework import DataQualityFramework
    from src.utils.logger import system_logger
    from src.core.decorators import handles_errors, validates, traced, log_execution_time
    import time
    import asyncio
    
    logger = system_logger.getChild('OptimisationPipeline')
    
    # Initialize comprehensive logging and progress tracking
    pipeline_start_time = time.time()
    step_count = 0
    total_steps = 2  # Confidence calibration + Parameter optimization
    error_count = 0
    warning_count = 0
    quality_issues = []
    
    def log_progress(step_name: str, step_number: int, total_steps: int, status: str = "STARTING"):
        """Log progress with emojis and detailed information."""
        progress_pct = (step_number / total_steps) * 100
        timestamp = format_datetime(get_current_datetime(), '%H:%M:%S')
        
        if status == "STARTING":
            logger.info(f"🚀 [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name}")
            print(f"🚀 [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name}")
        elif status == "COMPLETED":
            logger.info(f"✅ [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name} COMPLETED")
            print(f"✅ [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name} COMPLETED")
        elif status == "FAILED":
            logger.error(f"❌ [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name} FAILED")
            print(f"❌ [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name} FAILED")
        elif status == "WARNING":
            logger.warning(f"⚠️ [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name} WARNING")
            print(f"⚠️ [{timestamp}] STEP {step_number}/{total_steps} ({progress_pct:.1f}%) - {step_name} WARNING")
    
    def log_quality_issue(issue_type: str, description: str, severity: str = "WARNING"):
        """Log data quality issues with detailed reporting."""
        global quality_issues
        timestamp = format_datetime(get_current_datetime(), '%H:%M:%S')
        
        issue = {
            'timestamp': timestamp,
            'type': issue_type,
            'description': description,
            'severity': severity
        }
        quality_issues.append(issue)
        
        if severity == "ERROR":
            logger.error(f"🔴 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
            print(f"🔴 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
        elif severity == "WARNING":
            logger.warning(f"🟡 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
            print(f"🟡 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
        else:
            logger.info(f"🔵 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
            print(f"🔵 [{timestamp}] QUALITY ISSUE - {issue_type}: {description}")
    
    def log_performance_metrics(step_name: str, duration: float, memory_usage: float = None):
        """Log performance metrics for optimization steps."""
        timestamp = format_datetime(get_current_datetime(), '%H:%M:%S')
        logger.info(f"📊 [{timestamp}] PERFORMANCE - {step_name}: {duration:.2f}s")
        print(f"📊 [{timestamp}] PERFORMANCE - {step_name}: {duration:.2f}s")
        if memory_usage:
            logger.info(f"💾 [{timestamp}] MEMORY - {step_name}: {memory_usage:.2f}MB")
            print(f"💾 [{timestamp}] MEMORY - {step_name}: {memory_usage:.2f}MB")
    
    @handles_errors(fallback=False, context="optimisation_pipeline_step")
    @traced(span_name="optimisation_pipeline_step")
    async def execute_optimisation_step(step_name: str, step_func, *args, **kwargs):
        """Execute a single optimisation step with comprehensive error handling and monitoring."""
        nonlocal step_count, error_count, warning_count
        
        step_count += 1
        step_start_time = time.time()
        
        # Log step start with progress tracking
        log_progress(step_name, step_count, total_steps, "STARTING")
        logger.info(f"🔧 Executing {step_name} with args: {len(args)} args, {len(kwargs)} kwargs")
        
        try:
            # Execute the step function
            result = await step_func(*args, **kwargs)
            step_duration = time.time() - step_start_time
            
            # Log performance metrics
            log_performance_metrics(step_name, step_duration)
            
            if result:
                log_progress(step_name, step_count, total_steps, "COMPLETED")
                logger.info(f"🎉 {step_name} completed successfully in {step_duration:.2f}s")
                print(f"🎉 {step_name} completed successfully in {step_duration:.2f}s")
                return True
            else:
                error_count += 1
                log_progress(step_name, step_count, total_steps, "FAILED")
                logger.error(f"💥 {step_name} failed after {step_duration:.2f}s")
                print(f"💥 {step_name} failed after {step_duration:.2f}s")
                log_quality_issue("STEP_EXECUTION", f"{step_name} returned False", "ERROR")
                return False
                
        except Exception as e:
            error_count += 1
            step_duration = time.time() - step_start_time
            log_progress(step_name, step_count, total_steps, "FAILED")
            logger.exception(f"💥 {step_name} failed with exception after {step_duration:.2f}s: {e}")
            print(f"💥 {step_name} failed with exception after {step_duration:.2f}s: {e}")
            log_quality_issue("STEP_EXCEPTION", f"{step_name}: {str(e)}", "ERROR")
            return False
    
    @validates()
    async def validate_pipeline_data(symbol, exchange, data_dir):
        """Validate data quality before starting optimisation with comprehensive reporting."""
        nonlocal warning_count
        
        logger.info("🔍 Starting comprehensive pipeline data validation...")
        print("🔍 Starting comprehensive pipeline data validation...")
        
        dq_framework = DataQualityFramework()
        validation_start_time = time.time()
        validation_issues = []
        
        # Check data files
        data_files = [
            f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"{data_dir}/volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        logger.info(f"📁 Checking {len(data_files)} required data files...")
        print(f"📁 Checking {len(data_files)} required data files...")
        
        for i, file_path in enumerate(data_files, 1):
            logger.info(f"🔍 [{i}/{len(data_files)}] Validating: {file_path}")
            print(f"🔍 [{i}/{len(data_files)}] Validating: {file_path}")
            
            if not safe_file_exists(file_path):
                error_msg = f"Required data file missing: {file_path}"
                logger.error(f"❌ {error_msg}")
                print(f"❌ {error_msg}")
                log_quality_issue("MISSING_FILE", error_msg, "ERROR")
                validation_issues.append(error_msg)
                continue
            
            # File exists - check size and basic properties
            try:
                import os
                file_size = os.path.getsize(file_path)
                logger.info(f"📊 File size: {file_size / (1024*1024):.2f} MB")
                print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
                
                if file_size == 0:
                    error_msg = f"Data file is empty: {file_path}"
                    logger.error(f"❌ {error_msg}")
                    print(f"❌ {error_msg}")
                    log_quality_issue("EMPTY_FILE", error_msg, "ERROR")
                    validation_issues.append(error_msg)
                    continue
                
            except Exception as e:
                warning_msg = f"Could not check file size for {file_path}: {e}"
                logger.warning(f"⚠️ {warning_msg}")
                print(f"⚠️ {warning_msg}")
                log_quality_issue("FILE_SIZE_CHECK", warning_msg, "WARNING")
                warning_count += 1
            
            # Basic data quality check
            try:
                import pandas as pd
                logger.info(f"📖 Reading parquet file: {file_path}")
                print(f"📖 Reading parquet file: {file_path}")
                
                df = pd.read_parquet(file_path)
                
                if df.empty:
                    error_msg = f"Data file is empty: {file_path}"
                    logger.error(f"❌ {error_msg}")
                    print(f"❌ {error_msg}")
                    log_quality_issue("EMPTY_DATAFRAME", error_msg, "ERROR")
                    validation_issues.append(error_msg)
                    continue
                
                # Log basic dataframe info
                logger.info(f"📊 DataFrame shape: {df.shape}")
                print(f"📊 DataFrame shape: {df.shape}")
                logger.info(f"📊 DataFrame columns: {list(df.columns)}")
                print(f"📊 DataFrame columns: {list(df.columns)}")
                
                # Check for critical data quality issues
                logger.info(f"🔍 Running data quality validation...")
                print(f"🔍 Running data quality validation...")
                
                quality_result = dq_framework.validate_data(df, ['klines_schema'])
                
                if not quality_result.get('overall_passed', False):
                    warning_msg = f"Data quality issues in {file_path}: {quality_result.get('errors', [])}"
                    logger.warning(f"⚠️ {warning_msg}")
                    print(f"⚠️ {warning_msg}")
                    log_quality_issue("DATA_QUALITY", warning_msg, "WARNING")
                    warning_count += 1
                else:
                    logger.info(f"✅ Data quality validation passed for {file_path}")
                    print(f"✅ Data quality validation passed for {file_path}")
                
            except Exception as e:
                error_msg = f"Error reading data file {file_path}: {e}"
                logger.error(f"❌ {error_msg}")
                print(f"❌ {error_msg}")
                log_quality_issue("FILE_READ_ERROR", error_msg, "ERROR")
                validation_issues.append(error_msg)
                continue
        
        validation_duration = time.time() - validation_start_time
        log_performance_metrics("Data Validation", validation_duration)
        
        # Summary of validation results
        if validation_issues:
            logger.error(f"❌ Pipeline data validation failed with {len(validation_issues)} critical issues")
            print(f"❌ Pipeline data validation failed with {len(validation_issues)} critical issues")
            for issue in validation_issues:
                logger.error(f"   • {issue}")
                print(f"   • {issue}")
            return False
        else:
            logger.info(f"✅ Pipeline data validation passed in {validation_duration:.2f}s")
            print(f"✅ Pipeline data validation passed in {validation_duration:.2f}s")
            if warning_count > 0:
                logger.warning(f"⚠️ Found {warning_count} warnings during validation")
                print(f"⚠️ Found {warning_count} warnings during validation")
            return True
    
    try:
        # Pipeline initialization with comprehensive logging
        logger.info("🚀 Starting enhanced optimisation pipeline")
        logger.info(f"📊 Configuration: {config}")
        print("=" * 80)
        print("🚀 ENHANCED OPTIMISATION PIPELINE START")
        print("=" * 80)
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print(f"⏰ Timeframe: {timeframe}")
        print(f"📁 Data Directory: {data_dir}")
        print(f"⏰ Start Time: {format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Pre-pipeline validation
        if config.get('validation_enabled', True):
            logger.info("🔍 Starting pre-pipeline validation...")
            print("🔍 Starting pre-pipeline validation...")
            
            validation_success = await validate_pipeline_data(symbol, exchange, data_dir)
            if not validation_success:
                logger.error("❌ Pre-pipeline validation failed - cannot proceed")
                print("❌ Pre-pipeline validation failed - cannot proceed")
                log_quality_issue("PRE_VALIDATION", "Pre-pipeline validation failed", "ERROR")
                return False
            else:
                logger.info("✅ Pre-pipeline validation passed")
                print("✅ Pre-pipeline validation passed")
        
        # Initialize data quality framework for monitoring
        dq_framework = DataQualityFramework()
        logger.info("🛡️ Data quality framework initialized for monitoring")
        print("🛡️ Data quality framework initialized for monitoring")
        
        # Step 1: Confidence Calibration (if enabled)
        if config.get('confidence_calibration', True):
            logger.info("🎯 Starting Step 1: Confidence Calibration")
            print("🎯 Starting Step 1: Confidence Calibration")
            
            try:
                # Create enhanced confidence calibrator with data protection
                confidence_calibrator = ConfidenceCalibrationPerRegimeStep(config)
                logger.info("🔧 Confidence calibrator initialized")
                print("🔧 Confidence calibrator initialized")
                
                # Execute with comprehensive error handling
                step1_success = await execute_optimisation_step(
                    "Confidence Calibration",
                    confidence_calibrator.calibrate_confidence,
                    symbol, exchange, timeframe, data_dir
                )
                
                if not step1_success:
                    error_count += 1
                    logger.error("❌ Confidence calibration failed")
                    print("❌ Confidence calibration failed")
                    log_quality_issue("CONFIDENCE_CALIBRATION", "Confidence calibration step failed", "ERROR")
                    
                    if config.get('strict_mode', False):
                        logger.error("🛑 Strict mode enabled - stopping pipeline")
                        print("🛑 Strict mode enabled - stopping pipeline")
                        return False
                    else:
                        logger.warning("⚠️ Continuing with default confidence parameters")
                        print("⚠️ Continuing with default confidence parameters")
                        log_quality_issue("FALLBACK", "Using default confidence parameters", "WARNING")
                else:
                    logger.info("✅ Confidence calibration completed successfully")
                    print("✅ Confidence calibration completed successfully")
            except Exception as e:
                error_count += 1
                logger.exception(f"💥 Confidence calibration failed with exception: {e}")
                print(f"💥 Confidence calibration failed with exception: {e}")
                log_quality_issue("CONFIDENCE_CALIBRATION_EXCEPTION", str(e), "ERROR")
                
                if config.get('strict_mode', False):
                    return False
                else:
                    logger.warning("⚠️ Continuing with default confidence parameters")
                    print("⚠️ Continuing with default confidence parameters")
        
        # Step 2: Final Parameters Optimization (if enabled)
        if config.get('parameter_optimization', True):
            logger.info("🎯 Starting Step 2: Final Parameters Optimization")
            print("🎯 Starting Step 2: Final Parameters Optimization")
            
            try:
                # Create enhanced parameter optimizer with data protection
                param_optimizer = FinalParametersOptimizationStep(config)
                logger.info("🔧 Parameter optimizer initialized")
                print("🔧 Parameter optimizer initialized")
                
                # Execute with comprehensive error handling
                step2_success = await execute_optimisation_step(
                    "Final Parameters Optimization",
                    param_optimizer.optimize_parameters,
                    symbol, exchange, timeframe, data_dir
                )
                
                if not step2_success:
                    error_count += 1
                    logger.error("❌ Final parameters optimization failed")
                    print("❌ Final parameters optimization failed")
                    log_quality_issue("PARAMETER_OPTIMIZATION", "Parameter optimization step failed", "ERROR")
                    
                    if config.get('strict_mode', False):
                        logger.error("🛑 Strict mode enabled - stopping pipeline")
                        print("🛑 Strict mode enabled - stopping pipeline")
                        return False
                    else:
                        logger.warning("⚠️ Continuing with default parameters")
                        print("⚠️ Continuing with default parameters")
                        log_quality_issue("FALLBACK", "Using default parameters", "WARNING")
                else:
                    logger.info("✅ Final parameters optimization completed successfully")
                    print("✅ Final parameters optimization completed successfully")
            except Exception as e:
                error_count += 1
                logger.exception(f"💥 Final parameters optimization failed with exception: {e}")
                print(f"💥 Final parameters optimization failed with exception: {e}")
                log_quality_issue("PARAMETER_OPTIMIZATION_EXCEPTION", str(e), "ERROR")
                
                if config.get('strict_mode', False):
                    return False
                else:
                    logger.warning("⚠️ Continuing with default parameters")
                    print("⚠️ Continuing with default parameters")
        
        # Post-pipeline validation and cleanup
        if config.get('post_validation', True):
            logger.info("🔍 Performing post-pipeline validation...")
            print("🔍 Performing post-pipeline validation...")
            
            post_validation_start_time = time.time()
            
            # Check that output files were created
            expected_outputs = [
                f"models/{symbol}_{exchange}_confidence_calibration.json",
                f"models/{symbol}_{exchange}_optimized_parameters.json"
            ]
            
            missing_outputs = []
            created_outputs = []
            
            for output_file in expected_outputs:
                if not safe_file_exists(output_file):
                    missing_outputs.append(output_file)
                    log_quality_issue("MISSING_OUTPUT", f"Expected output file missing: {output_file}", "WARNING")
                else:
                    created_outputs.append(output_file)
                    logger.info(f"✅ Output file created: {output_file}")
                    print(f"✅ Output file created: {output_file}")
            
            if missing_outputs:
                warning_count += len(missing_outputs)
                logger.warning(f"⚠️ {len(missing_outputs)} output files missing: {missing_outputs}")
                print(f"⚠️ {len(missing_outputs)} output files missing: {missing_outputs}")
            else:
                logger.info("✅ All expected output files created")
                print("✅ All expected output files created")
            
            post_validation_duration = time.time() - post_validation_start_time
            log_performance_metrics("Post-Pipeline Validation", post_validation_duration)
        
        # Calculate total pipeline execution time
        total_pipeline_time = time.time() - pipeline_start_time
        
        # Save comprehensive pipeline execution summary
        if config.get('save_summary', True):
            logger.info("💾 Saving comprehensive pipeline execution summary...")
            print("💾 Saving comprehensive pipeline execution summary...")
            
            summary_file = f"{data_dir}/optimisation_pipeline_summary_{symbol}_{timeframe}.json"
            summary_data = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'config': config,
                'execution_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                'total_duration_seconds': total_pipeline_time,
                'success': error_count == 0,
                'error_count': error_count,
                'warning_count': warning_count,
                'quality_issues': quality_issues,
                'steps_completed': {
                    'confidence_calibration': config.get('confidence_calibration', True),
                    'parameter_optimization': config.get('parameter_optimization', True)
                },
                'output_files': {
                    'created': created_outputs if 'created_outputs' in locals() else [],
                    'missing': missing_outputs if 'missing_outputs' in locals() else []
                },
                'performance_metrics': {
                    'total_steps': total_steps,
                    'completed_steps': step_count,
                    'success_rate': (step_count - error_count) / step_count if step_count > 0 else 0
                }
            }
            
            safe_json_dump(summary_data, summary_file, indent=2)
            logger.info(f"💾 Comprehensive pipeline summary saved to: {summary_file}")
            print(f"💾 Comprehensive pipeline summary saved to: {summary_file}")
        
        # Final comprehensive reporting
        print("\n" + "=" * 80)
        print("📊 ENHANCED OPTIMISATION PIPELINE RESULTS")
        print("=" * 80)
        print(f"🎯 Symbol: {symbol}")
        print(f"🏢 Exchange: {exchange}")
        print(f"⏰ Timeframe: {timeframe}")
        print(f"⏱️ Total Duration: {total_pipeline_time:.2f} seconds")
        print(f"📈 Steps Completed: {step_count}/{total_steps}")
        print(f"❌ Errors: {error_count}")
        print(f"⚠️ Warnings: {warning_count}")
        print(f"🔍 Quality Issues: {len(quality_issues)}")
        
        if error_count == 0:
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("✅ All optimization steps completed with validation")
            logger.info("🎉 Enhanced optimisation pipeline completed successfully")
        else:
            print("⚠️ PIPELINE COMPLETED WITH ISSUES")
            print(f"❌ {error_count} errors encountered during execution")
            logger.warning(f"⚠️ Enhanced optimisation pipeline completed with {error_count} errors")
        
        if quality_issues:
            print("\n🔍 QUALITY ISSUES SUMMARY:")
            for issue in quality_issues:
                severity_emoji = "🔴" if issue['severity'] == "ERROR" else "🟡" if issue['severity'] == "WARNING" else "🔵"
                print(f"   {severity_emoji} [{issue['timestamp']}] {issue['type']}: {issue['description']}")
        
        print("=" * 80)
        
        return error_count == 0
        
    except Exception as e:
        total_pipeline_time = time.time() - pipeline_start_time
        error_count += 1
        
        logger.exception(f"❌ Enhanced optimisation pipeline failed: {e}")
        print(f"\n💥 ENHANCED OPTIMISATION PIPELINE FAILED!")
        print("=" * 80)
        print(f"⏱️ Duration before failure: {total_pipeline_time:.2f} seconds")
        print(f"❌ Error: {str(e)}")
        print("=" * 80)
        
        log_quality_issue("PIPELINE_EXCEPTION", str(e), "ERROR")
        
        # Save failure summary
        try:
            failure_summary_file = f"{data_dir}/optimisation_pipeline_failure_{symbol}_{timeframe}.json"
            failure_data = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'config': config,
                'failure_time': format_datetime(get_current_datetime(), '%Y-%m-%d %H:%M:%S'),
                'duration_before_failure': total_pipeline_time,
                'error': str(e),
                'error_type': type(e).__name__,
                'error_count': error_count,
                'warning_count': warning_count,
                'quality_issues': quality_issues
            }
            
            safe_json_dump(failure_data, failure_summary_file, indent=2)
            logger.info(f"💾 Failure summary saved to: {failure_summary_file}")
            print(f"💾 Failure summary saved to: {failure_summary_file}")
        except Exception as save_error:
            logger.warning(f"⚠️ Could not save failure summary: {save_error}")
        
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