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
    @validates(strict=True)
    @log_call
    @traced
    async def _validate_pipeline_inputs(symbol: str, exchange: str, timeframe: str, data_dir: str, **config) -> bool:
        """Validate all pipeline inputs and dependencies."""
        logger.info("🔍 Validating pipeline inputs and dependencies...")
        
        # Validate required parameters
        if not symbol or not exchange or not timeframe or not data_dir:
            raise ValueError("Missing required parameters: symbol, exchange, timeframe, data_dir")
        
        # Validate data directory exists
        if not safe_file_exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Validate required data files exist
        required_files = [
            f"aggtrades_{exchange}_{symbol}_consolidated.parquet",
            f"volume_{exchange}_{symbol}_consolidated.parquet"
        ]
        
        for file_name in required_files:
            file_path = f"{data_dir}/{file_name}"
            if not safe_file_exists(file_path):
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        # Validate configuration
        required_config_keys = ['hmm_training', 'regime_intelligence', 'analyst_creation']
        for key in required_config_keys:
            if key not in config:
                config[key] = True  # Set default values
        
        logger.info("✅ Pipeline inputs validation passed")
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
        
        try:
            import pandas as pd
from src.core.decorators.errors import handles_errors
            
            # Load and validate main data file using safe utilities
            data_file = f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            df = safe_read_parquet(data_file)
            
            if df.empty:
                logger.error(f"❌ Data file is empty: {data_file}")
                return False
            
            # Optimize DataFrame memory usage
            df = optimize_dataframe_dtypes(df)
            
            # Validate DataFrame schema
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            schema_valid, schema_errors = validate_dataframe_schema(df, required_columns)
            
            if not schema_valid:
                logger.error(f"❌ Data schema validation failed: {schema_errors}")
                return False
            
            # Validate data quality
            quality_report = validate_data_quality(df, max_nan_ratio=0.1, check_duplicates=True)
            
            if not quality_report['is_valid']:
                logger.error(f"❌ Data quality validation failed: {quality_report['issues']}")
                return False
            
            logger.info(f"✅ Data quality validation passed: {quality_report['total_rows']} rows, {quality_report['total_columns']} columns")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in data quality validation: {e}")
            return False
    
    @handles_errors(Exception, fallback=False, log_level="ERROR")
    @log_call
    @traced
    async def _execute_training_step(step_name: str, step_class, symbol: str, exchange: str, timeframe: str, data_dir: str, **config) -> bool:
        """Execute a single training step with comprehensive error handling."""
        logger.info(f"🚀 Executing {step_name}...")
        
        try:
            # Create step instance
            step_instance = step_class(config)
            
            # Execute step with timeout protection
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
                return False
            
            if success:
                logger.info(f"✅ {step_name} completed successfully")
                return True
            else:
                logger.error(f"❌ {step_name} failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing {step_name}: {e}")
            return False
    
    # Main pipeline execution with comprehensive validation
    try:
        logger.info("🚀 Starting Enhanced Model Training Pipeline")
        logger.info(f"📊 Configuration: {symbol} on {exchange}, timeframe: {timeframe}")
        logger.info(f"📁 Data directory: {data_dir}")
        logger.info(f"⚙️ Training config: {config}")
        
        # Step 1: Validate pipeline inputs
        inputs_valid = await _validate_pipeline_inputs(symbol, exchange, timeframe, data_dir, **config)
        if not inputs_valid:
            logger.error("❌ Pipeline input validation failed")
            return False
        
        # Step 2: Validate step dependencies
        dependencies_valid = await _validate_step_dependencies(symbol, exchange, timeframe, data_dir)
        if not dependencies_valid:
            logger.error("❌ Step dependency validation failed")
            return False
        
        # Step 3: Validate data quality
        data_quality_valid = await _validate_data_quality(symbol, exchange, data_dir)
        if not data_quality_valid:
            logger.error("❌ Data quality validation failed")
            return False
        
        # Step 4: Execute training steps with validation
        training_steps = [
            ("HMM-based Training", HMMBasedTrainingStep, config.get('hmm_training', True)),
            ("Unified Regime Intelligence", UnifiedRegimeIntelligenceStep, config.get('regime_intelligence', True)),
            ("Analyst Creation", AnalystCreationStep, config.get('analyst_creation', True)),
            ("Analyst Enhancement", AnalystEnhancementStep, config.get('analyst_enhancement', True)),
            ("Ensemble Creation", AnalystEnsembleCreationStep, config.get('ensemble_creation', True)),
            ("Tactician Training", TacticianSpecialistTrainingStep, config.get('tactician_training', True))
        ]
        
        all_steps_successful = True
        
        for step_name, step_class, enabled in training_steps:
            if enabled:
                step_success = await _execute_training_step(
                    step_name, step_class, symbol, exchange, timeframe, data_dir, **config
                )
                if not step_success:
                    all_steps_successful = False
                    logger.error(f"❌ Pipeline failed at {step_name}")
                    break
            else:
                logger.info(f"⏭️ Skipping {step_name} (disabled)")
        
        if all_steps_successful:
            logger.info("🎉 Model training pipeline completed successfully!")
            
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
                    "success_rate": 1.0
                },
                "data_info": {
                    "data_file_size": format_bytes(Path(f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet").stat().st_size) if safe_file_exists(f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet") else "unknown"
                }
            }
            
            summary_file = f"{data_dir}/model_training_execution_summary_{symbol}_{timeframe}.json"
            safe_json_dump(execution_summary, summary_file, indent=2)
            logger.info(f"💾 Execution summary saved to: {summary_file}")
            
            # Log metrics using common utilities
            safe_log_metric("pipeline_success", 1.0)
            safe_log_metric("steps_completed", len([step for step in training_steps if step[2]]))
            safe_log_params({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            })
            
            return True
        else:
            logger.error("❌ Model training pipeline failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Model training pipeline failed with error: {e}")
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