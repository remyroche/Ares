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

# Main pipeline function
async def run_model_training_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete model training pipeline."""
    try:
        # Step 1: HMM-based Training (if enabled)
        if config.get('hmm_training', True):
            hmm_trainer = HMMBasedTrainingStep()
            await hmm_trainer.train_models(symbol, exchange, timeframe, data_dir)
        
        # Step 2: Unified Regime Intelligence (if enabled)
        if config.get('regime_intelligence', True):
            regime_intelligence = UnifiedRegimeIntelligenceStep()
            await regime_intelligence.build_intelligence(symbol, exchange, timeframe, data_dir)
        
        # Step 3: Analyst Creation (if enabled)
        if config.get('analyst_creation', True):
            analyst_creator = AnalystCreationStep()
            await analyst_creator.create_analysts(symbol, exchange, timeframe, data_dir)
        
        # Step 4: Analyst Enhancement (if enabled)
        if config.get('analyst_enhancement', True):
            analyst_enhancer = AnalystEnhancementStep()
            await analyst_enhancer.enhance_analysts(symbol, exchange, timeframe, data_dir)
        
        # Step 5: Ensemble Creation (if enabled)
        if config.get('ensemble_creation', True):
            ensemble_creator = AnalystEnsembleCreationStep()
            await ensemble_creator.create_ensembles(symbol, exchange, timeframe, data_dir)
        
        # Step 6: Tactician Training (if enabled)
        if config.get('tactician_training', True):
            tactician_trainer = TacticianSpecialistTrainingStep()
            await tactician_trainer.train_tacticians(symbol, exchange, timeframe, data_dir)
        
        return True
        
    except Exception as e:
        print(f"Model training pipeline failed: {e}")
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