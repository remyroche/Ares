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

# Import core decorators for enhanced pipeline
from src.core.decorators import (
    handles_errors,
    retry,
    timeout,
    log_execution_time,
    traced,
    validates,
    validate_dataframe,
)
from src.utils.compat import handle_errors
from src.utils.logger import system_logger
from src.utils.pipeline_validation_utils import (
    pipeline_validator,
    validate_pipeline_step,
    get_pipeline_validation_summary,
)
from src.utils.common_operations import (
    validate_dataframe_integrity,
    validate_pipeline_step_output,
)

# Main pipeline function with enhanced validation and error handling
@handles_errors(
    fallback=False,
    log_level="ERROR",
    include_traceback=True
)
@retry(
    max_attempts=3,
    backoff_factor=2.0,
    exceptions=(ConnectionError, TimeoutError, ValueError)
)
@timeout(seconds=3600)  # 1 hour timeout
@log_execution_time
@traced
@validates(strict=True)
async def run_model_training_pipeline(
    symbol: str, 
    exchange: str, 
    timeframe: str, 
    data_dir: str, 
    **config
) -> bool:
    """Run the complete model training pipeline with enhanced validation and error handling."""
    logger = system_logger.getChild("ModelTrainingPipeline")
    
    logger.info(f"Starting model training pipeline for {symbol} on {exchange}")
    logger.info(f"Configuration: {config}")
    
    try:
        # Pre-pipeline validation
        logger.info("Starting pre-pipeline validation")
        data_loading_validation = await validate_pipeline_step(
            "data_loading",
            None,
            "data_loading",
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir
        )
        
        if not data_loading_validation.get('is_valid', False):
            logger.error("Pre-pipeline validation failed")
            return False
        
        # Step 1: HMM-based Training (if enabled)
        if config.get('hmm_training', True):
            logger.info("Step 1: Starting HMM-based training")
            hmm_trainer = HMMBasedTrainingStep()
            hmm_result = await hmm_trainer.train_models(symbol, exchange, timeframe, data_dir)
            
            # Validate HMM training output
            hmm_validation = await validate_pipeline_step(
                "hmm_training",
                hmm_result,
                "model_training_output",
                expected_metrics=['accuracy', 'loss', 'convergence_iterations']
            )
            
            if not hmm_validation.get('is_valid', False):
                logger.error("HMM training validation failed")
                return False
            
            logger.info("Step 1: HMM-based training completed and validated")
        
        # Step 2: Unified Regime Intelligence (if enabled)
        if config.get('regime_intelligence', True):
            logger.info("Step 2: Starting unified regime intelligence")
            regime_intelligence = UnifiedRegimeIntelligenceStep()
            regime_result = await regime_intelligence.build_intelligence(symbol, exchange, timeframe, data_dir)
            
            # Validate regime intelligence output
            regime_validation = await validate_pipeline_step(
                "regime_intelligence",
                regime_result,
                "model_training_output",
                expected_metrics=['regime_accuracy', 'transition_accuracy', 'confidence_score']
            )
            
            if not regime_validation.get('is_valid', False):
                logger.error("Regime intelligence validation failed")
                return False
            
            logger.info("Step 2: Unified regime intelligence completed and validated")
        
        # Step 3: Analyst Creation (if enabled)
        if config.get('analyst_creation', True):
            logger.info("Step 3: Starting analyst creation")
            analyst_creator = AnalystCreationStep()
            analyst_result = await analyst_creator.create_analysts(symbol, exchange, timeframe, data_dir)
            
            # Validate analyst creation output
            analyst_validation = await validate_pipeline_step(
                "analyst_creation",
                analyst_result,
                "model_training_output",
                expected_metrics=['creation_accuracy', 'model_count']
            )
            
            if not analyst_validation.get('is_valid', False):
                logger.error("Analyst creation validation failed")
                return False
            
            logger.info("Step 3: Analyst creation completed and validated")
        
        # Step 4: Analyst Enhancement (if enabled)
        if config.get('analyst_enhancement', True):
            logger.info("Step 4: Starting analyst enhancement")
            analyst_enhancer = AnalystEnhancementStep()
            enhancement_result = await analyst_enhancer.enhance_analysts(symbol, exchange, timeframe, data_dir)
            
            # Validate analyst enhancement output
            enhancement_validation = await validate_pipeline_step(
                "analyst_enhancement",
                enhancement_result,
                "model_training_output",
                expected_metrics=['enhancement_accuracy', 'improvement_scores']
            )
            
            if not enhancement_validation.get('is_valid', False):
                logger.error("Analyst enhancement validation failed")
                return False
            
            logger.info("Step 4: Analyst enhancement completed and validated")
        
        # Step 5: Ensemble Creation (if enabled)
        if config.get('ensemble_creation', True):
            logger.info("Step 5: Starting ensemble creation")
            ensemble_creator = AnalystEnsembleCreationStep()
            ensemble_result = await ensemble_creator.create_ensembles(symbol, exchange, timeframe, data_dir)
            
            # Validate ensemble creation output
            ensemble_validation = await validate_pipeline_step(
                "ensemble_creation",
                ensemble_result,
                "model_training_output",
                expected_metrics=['ensemble_accuracy', 'ensemble_count']
            )
            
            if not ensemble_validation.get('is_valid', False):
                logger.error("Ensemble creation validation failed")
                return False
            
            logger.info("Step 5: Ensemble creation completed and validated")
        
        # Step 6: Tactician Training (if enabled)
        if config.get('tactician_training', True):
            logger.info("Step 6: Starting tactician training")
            tactician_trainer = TacticianSpecialistTrainingStep()
            tactician_result = await tactician_trainer.train_tacticians(symbol, exchange, timeframe, data_dir)
            
            # Validate tactician training output
            tactician_validation = await validate_pipeline_step(
                "tactician_training",
                tactician_result,
                "model_training_output",
                expected_metrics=['accuracy', 'precision', 'recall', 'f1_score']
            )
            
            if not tactician_validation.get('is_valid', False):
                logger.error("Tactician training validation failed")
                return False
            
            logger.info("Step 6: Tactician training completed and validated")
        
        # Get final validation summary
        validation_summary = get_pipeline_validation_summary()
        logger.info(f"Pipeline validation summary: {validation_summary['success_rate']:.2%} success rate")
        
        logger.info("Model training pipeline completed successfully with full validation")
        return True
        
    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}")
        raise

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