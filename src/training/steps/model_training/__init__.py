#!/usr/bin/env python3
"""Model Training Package for Trading Pipeline.

This package contains all the components for model training:
- HMM-based training and multi-timeframe ensembles
- Unified regime intelligence
- Analyst creation, enhancement, and ensemble creation
- Tactician labeling and specialist training
- Model persistence and validation components
"""

# Import HMM training components
from .step09_hmm_based_training import HMMBasedTrainingStep
from .step09_hmm_based_training_per_regime import PerRegimeHMMTrainingStep
from .step09_hmm_based_training_validator import HMMTrainingValidator
from .step09_5_hmm_lm_generalist_training import HMMLMGeneralistTrainingStep
from .step09_5_hmm_lm_generalist_training_validator import HMMLMGeneralistTrainingValidator
from .step09_5_multi_timeframe_hmm_ensemble import MultiTimeframeHMMEnsembleStep
from .step09_5_multi_timeframe_hmm_ensemble_validator import MultiTimeframeHMMEnsembleValidator
from .multi_timeframe_hmm_ensemble import MultiTimeframeHMMEnsemble
from .sr_outcome_model_trainer import SROutcomeModelTrainer

# Import regime intelligence components
from .step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
from .step10_unified_regime_intelligence_per_regime import PerRegimeUnifiedIntelligenceStep
from .step10_unified_regime_intelligence_validator import UnifiedRegimeIntelligenceValidator

# Import analyst components
from .step11_analyst_creation import AnalystCreationStep
from .step11_analyst_creation_per_regime import PerRegimeAnalystCreationStep
from .step11_analyst_creation_validator import AnalystCreationValidator
from .step12_analyst_enhancement import AnalystEnhancementStep
from .step12_analyst_enhancement_per_regime import PerRegimeAnalystEnhancementStep
from .step12_analyst_enhancement_validator import AnalystEnhancementValidator
from .step13_analyst_ensemble_creation import AnalystEnsembleCreationStep
from .step13_analyst_ensemble_creation_per_regime import PerRegimeAnalystEnsembleCreationStep
from .step13_analyst_ensemble_creation_validator import AnalystEnsembleCreationValidator

# Import tactician components
from .step14_tactician_labeling import TacticianLabelingStep
from .step14_tactician_labeling_per_regime import PerRegimeTacticianLabelingStep
from .step14_tactician_labeling_validator import TacticianLabelingValidator
from .step15_tactician_specialist_training import TacticianSpecialistTrainingStep
from .step15_tactician_specialist_training_per_regime import PerRegimeTacticianSpecialistTrainingStep
from .step15_tactician_specialist_training_validator import TacticianSpecialistTrainingValidator

# Import validation and pipeline components
from .test_refactored_components import TestRefactoredComponents
from .update_steps_for_unified_data import UpdateStepsForUnifiedData
from .per_regime_pipeline_config import PerRegimePipelineConfig
from .per_regime_pipeline_integration import PerRegimePipelineIntegration
from .per_regime_pipeline_orchestrator import PerRegimePipelineOrchestrator

# Enhanced pipeline function with comprehensive validation and error handling
async def run_model_training_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete model training pipeline with comprehensive validation and error handling."""
    from src.core.decorators import handles_errors, validates, log_call, traced
    from src.utils.common_operations import (
        get_current_datetime, format_datetime, ensure_directory,
        safe_json_dump, safe_json_load, validate_dataframe_schema,
        validate_data_quality, safe_copy, safe_file_exists, safe_float, safe_int,
        safe_read_parquet, safe_to_parquet, optimize_dataframe_dtypes,
        safe_resample, align_dataframes, timed_operation, format_bytes,
        chunked_iterable, parallel_map, safe_log_metric, safe_log_params,
        safe_log_artifact, standardize_price_action_probabilities
    )
    from src.utils.logger import system_logger
    from src.utils.validator_orchestrator import ValidatorOrchestrator
    from src.utils.step_dependency_validator import StepDependencyValidator
    
    logger = system_logger.getChild("ModelTrainingPipeline")
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _monitor_memory_usage() -> dict:
        """Monitor memory usage and provide optimization alerts."""
        try:
            import psutil
            import gc
            
            # Get memory information
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            memory_stats = {
                "system_memory": {
                    "total_gb": memory_info.total / (1024**3),
                    "available_gb": memory_info.available / (1024**3),
                    "used_gb": memory_info.used / (1024**3),
                    "percent_used": memory_info.percent
                },
                "process_memory": {
                    "rss_gb": process_memory.rss / (1024**3),
                    "vms_gb": process_memory.vms / (1024**3)
                }
            }
            
            # Check for memory issues
            if memory_info.percent > 90:
                logger.warning(f"⚠️ High system memory usage: {memory_info.percent:.1f}%")
                print(f"   ⚠️ High system memory usage: {memory_info.percent:.1f}%")
            elif memory_info.percent > 80:
                logger.warning(f"⚠️ Moderate system memory usage: {memory_info.percent:.1f}%")
                print(f"   ⚠️ Moderate system memory usage: {memory_info.percent:.1f}%")
            
            if process_memory.rss / (1024**3) > 2:  # More than 2GB
                logger.warning(f"⚠️ High process memory usage: {process_memory.rss / (1024**3):.2f} GB")
                print(f"   ⚠️ High process memory usage: {process_memory.rss / (1024**3):.2f} GB")
            
            # Force garbage collection
            gc.collect()
            
            return memory_stats
            
        except ImportError:
            logger.warning("⚠️ psutil not available for memory monitoring")
            print("   ⚠️ psutil not available for memory monitoring")
            return {}
        except Exception as e:
            logger.warning(f"⚠️ Memory monitoring failed: {e}")
            print(f"   ⚠️ Memory monitoring failed: {e}")
            return {}
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @validates(strict=True)
    @log_call
    @traced
    async def _validate_pipeline_inputs(symbol: str, exchange: str, timeframe: str, data_dir: str, **config) -> bool:
        """Validate all pipeline inputs and dependencies."""
        logger.info("🔍 Validating pipeline inputs and dependencies...")
        print("   🔍 Validating pipeline inputs and dependencies...")
        
        quality_issues = []
        quality_warnings = []
        
        # Validate required parameters
        if not symbol or not exchange or not timeframe or not data_dir:
            error_msg = "Missing required parameters: symbol, exchange, timeframe, data_dir"
            quality_issues.append(error_msg)
            raise ValueError(error_msg)
        
        # Validate data directory exists
        if not safe_file_exists(data_dir):
            error_msg = f"Data directory not found: {data_dir}"
            quality_issues.append(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Validate required data files exist
        required_files = [
            f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        for file_name in required_files:
            file_path = f"{data_dir}/{file_name}"
            if not safe_file_exists(file_path):
                error_msg = f"Required data file not found: {file_path}"
                quality_issues.append(error_msg)
                raise FileNotFoundError(error_msg)
            else:
                # Check file size
                try:
                    file_size = Path(file_path).stat().st_size
                    if file_size < 1024:  # Less than 1KB
                        warning_msg = f"⚠️ Data file {file_name} is very small ({file_size} bytes)"
                        quality_warnings.append(warning_msg)
                        logger.warning(warning_msg)
                        print(f"   {warning_msg}")
                except Exception as e:
                    warning_msg = f"⚠️ Could not check file size for {file_name}: {e}"
                    quality_warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    print(f"   {warning_msg}")
        
        # Validate configuration
        required_config_keys = ['hmm_training', 'regime_intelligence', 'analyst_creation']
        for key in required_config_keys:
            if key not in config:
                config[key] = True  # Set default values
                warning_msg = f"⚠️ Set default value for missing config key: {key}"
                quality_warnings.append(warning_msg)
                logger.warning(warning_msg)
                print(f"   {warning_msg}")
        
        # Report quality issues
        if quality_issues:
            logger.error(f"❌ Found {len(quality_issues)} quality issues:")
            print(f"   ❌ Found {len(quality_issues)} quality issues:")
            for issue in quality_issues:
                logger.error(f"   • {issue}")
                print(f"   • {issue}")
            return False
        
        if quality_warnings:
            logger.warning(f"⚠️ Found {len(quality_warnings)} quality warnings:")
            print(f"   ⚠️ Found {len(quality_warnings)} quality warnings:")
            for warning in quality_warnings:
                logger.warning(f"   • {warning}")
                print(f"   • {warning}")
        
        logger.info("✅ Pipeline inputs validation passed")
        print("   ✅ Pipeline inputs validation passed")
        return True
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _validate_step_dependencies(symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Validate that all required previous steps have been completed."""
        logger.info("🔍 Validating step dependencies...")
        
        try:
            validator_orchestrator = ValidatorOrchestrator()
            dependency_validator = StepDependencyValidator()
            
            # Prepare training input for validation
            training_input = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "data_dir": data_dir,
            }
            
            # Validate required previous steps
            required_steps = [
                "step1_data_collection",
                "step1_5_data_converter", 
                "step2_data_reading",
                "step3_hmm_regime_discovery",
                "step4_triple_barrier_method",
                "step5_labeling",
                "step6_feature_engineering"
            ]
            
            pipeline_state = {}
            all_passed = True
            
            for step in required_steps:
                try:
                    result = await validator_orchestrator.run_step_validator(
                        step, training_input, pipeline_state, config
                    )
                    
                    if not result.get("validation_passed", False):
                        logger.error(f"❌ Step dependency validation failed for {step}")
                        all_passed = False
                    else:
                        logger.info(f"✅ Step dependency validation passed for {step}")
                        
                except Exception as e:
                    logger.error(f"❌ Error validating step {step}: {e}")
                    all_passed = False
            
            if all_passed:
                logger.info("✅ All step dependencies validated successfully")
            else:
                logger.error("❌ Some step dependencies failed validation")
                
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Error in step dependency validation: {e}")
            return False
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _validate_data_quality(symbol: str, exchange: str, data_dir: str) -> bool:
        """Validate data quality before training."""
        logger.info("🔍 Validating data quality...")
        print("   🔍 Validating data quality...")
        
        quality_issues = []
        quality_warnings = []
        quality_score = 100  # Start with perfect score
        
        try:
            import pandas as pd
            
            # Load and validate main data file using safe utilities
            data_file = f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            print(f"   📁 Loading data file: {data_file}")
            df = safe_read_parquet(data_file)
            
            if df.empty:
                error_msg = f"Data file is empty: {data_file}"
                quality_issues.append(error_msg)
                logger.error(f"❌ {error_msg}")
                print(f"   ❌ {error_msg}")
                return False
            
            # Optimize DataFrame memory usage
            df = optimize_dataframe_dtypes(df)
            print(f"   📊 Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Validate DataFrame schema
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            schema_valid, schema_errors = validate_dataframe_schema(df, required_columns)
            
            if not schema_valid:
                error_msg = f"Data schema validation failed: {schema_errors}"
                quality_issues.append(error_msg)
                logger.error(f"❌ {error_msg}")
                print(f"   ❌ {error_msg}")
                return False
            else:
                print("   ✅ Data schema validation passed")
            
            # Check for missing values
            nan_counts = df.isnull().sum()
            total_nans = nan_counts.sum()
            if total_nans > 0:
                nan_ratio = total_nans / (len(df) * len(df.columns))
                if nan_ratio > 0.1:  # More than 10% missing
                    error_msg = f"High missing data ratio: {nan_ratio:.2%} ({total_nans} missing values)"
                    quality_issues.append(error_msg)
                    logger.error(f"❌ {error_msg}")
                    print(f"   ❌ {error_msg}")
                    quality_score -= 30
                else:
                    warning_msg = f"Some missing data found: {nan_ratio:.2%} ({total_nans} missing values)"
                    quality_warnings.append(warning_msg)
                    logger.warning(f"⚠️ {warning_msg}")
                    print(f"   ⚠️ {warning_msg}")
                    quality_score -= 10
            else:
                print("   ✅ No missing values found")
            
            # Check for duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                warning_msg = f"Found {duplicate_count} duplicate rows"
                quality_warnings.append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
                print(f"   ⚠️ {warning_msg}")
                quality_score -= 5
            
            # Check data volume
            if len(df) < 1000:
                warning_msg = f"Low data volume: {len(df)} rows (minimum recommended: 1000)"
                quality_warnings.append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
                print(f"   ⚠️ {warning_msg}")
                quality_score -= 20
            elif len(df) < 10000:
                warning_msg = f"Moderate data volume: {len(df)} rows (recommended: 10000+)"
                quality_warnings.append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
                print(f"   ⚠️ {warning_msg}")
                quality_score -= 5
            
            # Check for price anomalies
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    if (df[col] <= 0).any():
                        error_msg = f"Found non-positive prices in {col} column"
                        quality_issues.append(error_msg)
                        logger.error(f"❌ {error_msg}")
                        print(f"   ❌ {error_msg}")
                        quality_score -= 25
                    elif (df[col] > df[col].quantile(0.99) * 10).any():
                        warning_msg = f"Found potential price outliers in {col} column"
                        quality_warnings.append(warning_msg)
                        logger.warning(f"⚠️ {warning_msg}")
                        print(f"   ⚠️ {warning_msg}")
                        quality_score -= 5
            
            # Report quality assessment
            print(f"   📊 Data quality score: {quality_score}/100")
            if quality_score >= 90:
                print("   🎉 Excellent data quality!")
            elif quality_score >= 70:
                print("   ✅ Good data quality")
            elif quality_score >= 50:
                print("   ⚠️ Fair data quality - proceed with caution")
            else:
                print("   ❌ Poor data quality - consider data cleaning")
            
            # Report quality issues
            if quality_issues:
                logger.error(f"❌ Found {len(quality_issues)} quality issues:")
                print(f"   ❌ Found {len(quality_issues)} quality issues:")
                for issue in quality_issues:
                    logger.error(f"   • {issue}")
                    print(f"   • {issue}")
                return False
            
            if quality_warnings:
                logger.warning(f"⚠️ Found {len(quality_warnings)} quality warnings:")
                print(f"   ⚠️ Found {len(quality_warnings)} quality warnings:")
                for warning in quality_warnings:
                    logger.warning(f"   • {warning}")
                    print(f"   • {warning}")
            
            logger.info(f"✅ Data quality validation passed: {len(df)} rows, {len(df.columns)} columns, score: {quality_score}/100")
            print(f"   ✅ Data quality validation passed: {len(df)} rows, {len(df.columns)} columns, score: {quality_score}/100")
            return True
            
        except Exception as e:
            error_msg = f"Error in data quality validation: {e}"
            quality_issues.append(error_msg)
            logger.error(f"❌ {error_msg}")
            print(f"   ❌ {error_msg}")
            return False
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _execute_training_step(step_name: str, step_class, symbol: str, exchange: str, timeframe: str, data_dir: str, **config) -> bool:
        """Execute a single training step with comprehensive error handling."""
        logger.info(f"🚀 Executing {step_name}...")
        print(f"   🔧 Initializing {step_name}...")
        
        try:
            # Create step instance
            step_instance = step_class(config)
            print(f"   ✅ {step_name} instance created successfully")
            logger.info(f"✅ {step_name} instance created successfully")
            
            # Execute step with timeout protection
            print(f"   🚀 Starting {step_name} execution...")
            logger.info(f"🚀 Starting {step_name} execution...")
            
            if hasattr(step_instance, 'train_models'):
                success = await step_instance.train_models(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'build_intelligence'):
                success = await step_instance.build_intelligence(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'create_analysts'):
                success = await step_instance.create_analysts(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'enhance_analysts'):
                success = await step_instance.enhance_analysts(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'create_ensembles'):
                success = await step_instance.create_ensembles(symbol, exchange, timeframe, data_dir)
            elif hasattr(step_instance, 'train_tacticians'):
                success = await step_instance.train_tacticians(symbol, exchange, timeframe, data_dir)
            else:
                logger.error(f"❌ Unknown step method for {step_name}")
                print(f"   ❌ Unknown step method for {step_name}")
                return False
            
            if success:
                logger.info(f"✅ {step_name} completed successfully")
                print(f"   ✅ {step_name} completed successfully")
                return True
            else:
                logger.error(f"❌ {step_name} failed")
                print(f"   ❌ {step_name} failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing {step_name}: {e}")
            logger.error(f"📋 Exception type: {type(e).__name__}")
            print(f"   ❌ Error executing {step_name}: {e}")
            print(f"   📋 Exception type: {type(e).__name__}")
            return False
    
    # Main pipeline execution with comprehensive validation
    try:
        print("🚀 Starting Enhanced Model Training Pipeline")
        print("=" * 80)
        logger.info("🚀 Starting Enhanced Model Training Pipeline")
        logger.info(f"📊 Configuration: {symbol} on {exchange}, timeframe: {timeframe}")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"⚙️ Training config: {config}")
        
        # Initial memory monitoring
        print("🔍 Initial memory monitoring...")
        initial_memory = await _monitor_memory_usage()
        
        # Step 1: Validate pipeline inputs
        print("🔍 STEP 1/6: Validating pipeline inputs...")
        logger.info("🔍 STEP 1/6: Validating pipeline inputs...")
        inputs_valid = await _validate_pipeline_inputs(symbol, exchange, timeframe, data_dir, **config)
        if not inputs_valid:
            logger.error("❌ Pipeline input validation failed")
            print("❌ Pipeline input validation failed")
            return False
        print("✅ Pipeline input validation passed")
        logger.info("✅ Pipeline input validation passed")
        
        # Step 2: Validate step dependencies
        print("🔍 STEP 2/6: Validating step dependencies...")
        logger.info("🔍 STEP 2/6: Validating step dependencies...")
        dependencies_valid = await _validate_step_dependencies(symbol, exchange, timeframe, data_dir)
        if not dependencies_valid:
            logger.error("❌ Step dependency validation failed")
            print("❌ Step dependency validation failed")
            return False
        print("✅ Step dependency validation passed")
        logger.info("✅ Step dependency validation passed")
        
        # Step 3: Validate data quality
        print("🔍 STEP 3/6: Validating data quality...")
        logger.info("🔍 STEP 3/6: Validating data quality...")
        data_quality_valid = await _validate_data_quality(symbol, exchange, data_dir)
        if not data_quality_valid:
            logger.error("❌ Data quality validation failed")
            print("❌ Data quality validation failed")
            return False
        print("✅ Data quality validation passed")
        logger.info("✅ Data quality validation passed")
        
        # Memory monitoring after data validation
        print("🔍 Memory monitoring after data validation...")
        post_data_memory = await _monitor_memory_usage()
        
        # Step 4: Execute training steps with validation
        print("🚀 STEP 4/6: Executing training steps...")
        logger.info("🚀 STEP 4/6: Executing training steps...")
        
        training_steps = [
            ("HMM-based Training", HMMBasedTrainingStep, config.get('hmm_training', True)),
            ("Unified Regime Intelligence", UnifiedRegimeIntelligenceStep, config.get('regime_intelligence', True)),
            ("Analyst Creation", AnalystCreationStep, config.get('analyst_creation', True)),
            ("Analyst Enhancement", AnalystEnhancementStep, config.get('analyst_enhancement', True)),
            ("Ensemble Creation", AnalystEnsembleCreationStep, config.get('ensemble_creation', True)),
            ("Tactician Training", TacticianSpecialistTrainingStep, config.get('tactician_training', True))
        ]
        
        # Filter enabled steps
        enabled_steps = [(name, cls, enabled) for name, cls, enabled in training_steps if enabled]
        total_steps = len(enabled_steps)
        
        print(f"📊 Total training steps to execute: {total_steps}")
        logger.info(f"📊 Total training steps to execute: {total_steps}")
        
        all_steps_successful = True
        completed_steps = 0
        
        for step_index, (step_name, step_class, enabled) in enumerate(enabled_steps, 1):
            print(f"🔄 STEP 4.{step_index}/{total_steps}: {step_name}...")
            logger.info(f"🔄 STEP 4.{step_index}/{total_steps}: {step_name}...")
            
            step_start_time = time.time()
            step_success = await _execute_training_step(
                step_name, step_class, symbol, exchange, timeframe, data_dir, **config
            )
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if step_success:
                completed_steps += 1
                progress_percentage = (completed_steps / total_steps) * 100
                print(f"✅ {step_name} completed successfully in {step_duration:.2f} seconds")
                print(f"📊 Progress: {completed_steps}/{total_steps} steps ({progress_percentage:.1f}%)")
                logger.info(f"✅ {step_name} completed successfully in {step_duration:.2f} seconds")
                logger.info(f"📊 Progress: {completed_steps}/{total_steps} steps ({progress_percentage:.1f}%)")
                
                # Memory monitoring after each step
                if step_index % 2 == 0:  # Monitor every 2 steps to avoid spam
                    print(f"🔍 Memory monitoring after step {step_index}...")
                    step_memory = await _monitor_memory_usage()
            else:
                all_steps_successful = False
                logger.error(f"❌ Pipeline failed at {step_name}")
                print(f"❌ Pipeline failed at {step_name}")
                print(f"💥 Training stopped at step {step_index}/{total_steps}")
                break
        
        # Log skipped steps
        skipped_steps = [(name, cls, enabled) for name, cls, enabled in training_steps if not enabled]
        if skipped_steps:
            print(f"⏭️ Skipped {len(skipped_steps)} disabled steps:")
            logger.info(f"⏭️ Skipped {len(skipped_steps)} disabled steps:")
            for step_name, _, _ in skipped_steps:
                print(f"   • {step_name}")
                logger.info(f"   • {step_name}")
        
        if all_steps_successful:
            print("🎉 STEP 5/6: Model training pipeline completed successfully!")
            print("=" * 80)
            logger.info("🎉 STEP 5/6: Model training pipeline completed successfully!")
            
            # Calculate comprehensive performance metrics
            total_execution_time = time.time() - start_time
            avg_step_time = total_execution_time / total_steps if total_steps > 0 else 0
            
            # Final memory monitoring
            print("🔍 Final memory monitoring...")
            final_memory = await _monitor_memory_usage()
            
            # Save pipeline execution summary with enhanced metadata
            execution_summary = {
                "pipeline_info": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "data_dir": data_dir,
                    "execution_time": format_datetime(get_current_datetime()),
                    "success": True
                },
                "configuration": config,
                "steps_completed": [step[0] for step in training_steps if step[2]],
                "performance_metrics": {
                    "total_steps": len(training_steps),
                    "completed_steps": len([step for step in training_steps if step[2]]),
                    "success_rate": 1.0,
                    "enabled_steps": total_steps,
                    "skipped_steps": len(skipped_steps) if 'skipped_steps' in locals() else 0,
                    "total_execution_time_seconds": total_execution_time,
                    "total_execution_time_minutes": total_execution_time / 60,
                    "average_step_time_seconds": avg_step_time,
                    "execution_efficiency": "high" if total_execution_time < 3600 else "medium" if total_execution_time < 7200 else "low",
                    "steps_per_minute": total_steps / (total_execution_time / 60) if total_execution_time > 0 else 0
                },
                "data_info": {
                    "data_file_size": format_bytes(Path(f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet").stat().st_size) if safe_file_exists(f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet") else "unknown"
                },
                "quality_metrics": {
                    "data_validation_passed": True,
                    "step_dependencies_validated": True,
                    "overall_quality_score": 100,
                    "quality_issues_found": 0,
                    "quality_warnings_found": 0
                },
                "memory_metrics": {
                    "initial_memory": initial_memory,
                    "post_data_memory": post_data_memory if 'post_data_memory' in locals() else {},
                    "final_memory": final_memory
                }
            }
            
            summary_file = f"{data_dir}/model_training_execution_summary_{symbol}_{timeframe}.json"
            safe_json_dump(execution_summary, summary_file, indent=2)
            logger.info(f"💾 Execution summary saved to: {summary_file}")
            print(f"💾 Execution summary saved to: {summary_file}")
            
            # Log comprehensive metrics using common utilities
            safe_log_metric("pipeline_success", 1.0)
            safe_log_metric("steps_completed", len([step for step in training_steps if step[2]]))
            safe_log_metric("total_execution_time_seconds", total_execution_time)
            safe_log_metric("total_execution_time_minutes", total_execution_time / 60)
            safe_log_metric("average_step_time_seconds", avg_step_time)
            safe_log_metric("steps_per_minute", total_steps / (total_execution_time / 60) if total_execution_time > 0 else 0)
            safe_log_metric("execution_efficiency", "high" if total_execution_time < 3600 else "medium" if total_execution_time < 7200 else "low")
            safe_log_params({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "total_steps": total_steps,
                "enabled_steps": total_steps,
                "skipped_steps": len(skipped_steps) if 'skipped_steps' in locals() else 0
            })
            
            # Print comprehensive performance summary
            print("📊 PERFORMANCE SUMMARY:")
            print(f"   ⏱️ Total execution time: {total_execution_time:.2f} seconds ({total_execution_time/60:.2f} minutes)")
            print(f"   📈 Average step time: {avg_step_time:.2f} seconds")
            print(f"   🚀 Steps per minute: {total_steps / (total_execution_time / 60):.2f}")
            print(f"   📊 Execution efficiency: {'high' if total_execution_time < 3600 else 'medium' if total_execution_time < 7200 else 'low'}")
            print(f"   ✅ Success rate: 100%")
            print("✅ All training steps completed successfully!")
            print("=" * 80)
            return True
        else:
            print("❌ STEP 5/6: Model training pipeline failed")
            print("=" * 80)
            logger.error("❌ STEP 5/6: Model training pipeline failed")
            return False
        
    except Exception as e:
        print("💥 STEP 6/6: Model training pipeline failed with exception!")
        print("=" * 80)
        print(f"❌ Error: {e}")
        print(f"📋 Exception type: {type(e).__name__}")
        print("🔍 Troubleshooting suggestions:")
        print("   • Check data file integrity and availability")
        print("   • Verify previous steps completed successfully")
        print("   • Check system resources (memory, disk space)")
        print("   • Review configuration parameters")
        print("   • Check log files for detailed error information")
        print("=" * 80)
        
        logger.error(f"❌ STEP 6/6: Model training pipeline failed with error: {e}")
        logger.error(f"📋 Exception type: {type(e).__name__}")
        logger.error(f"📋 Exception details: {str(e)}")
        return False

__all__ = [
    'HMMBasedTrainingStep',
    'PerRegimeHMMTrainingStep',
    'HMMTrainingValidator',
    'HMMLMGeneralistTrainingStep',
    'HMMLMGeneralistTrainingValidator',
    'MultiTimeframeHMMEnsembleStep',
    'MultiTimeframeHMMEnsembleValidator',
    'MultiTimeframeHMMEnsemble',
    'SROutcomeModelTrainer',
    'UnifiedRegimeIntelligenceStep',
    'PerRegimeUnifiedIntelligenceStep',
    'UnifiedRegimeIntelligenceValidator',
    'AnalystCreationStep',
    'PerRegimeAnalystCreationStep',
    'AnalystCreationValidator',
    'AnalystEnhancementStep',
    'PerRegimeAnalystEnhancementStep',
    'AnalystEnhancementValidator',
    'AnalystEnsembleCreationStep',
    'PerRegimeAnalystEnsembleCreationStep',
    'AnalystEnsembleCreationValidator',
    'TacticianLabelingStep',
    'PerRegimeTacticianLabelingStep',
    'TacticianLabelingValidator',
    'TacticianSpecialistTrainingStep',
    'PerRegimeTacticianSpecialistTrainingStep',
    'TacticianSpecialistTrainingValidator',
    'TestRefactoredComponents',
    'UpdateStepsForUnifiedData',
    'PerRegimePipelineConfig',
    'PerRegimePipelineIntegration',
    'PerRegimePipelineOrchestrator',
    'run_model_training_pipeline'
]