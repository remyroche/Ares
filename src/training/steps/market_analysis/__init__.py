#!/usr/bin/env python3
"""Market Analysis Package for Trading Pipeline.

This package contains all the components for market analysis:
- HMM regime discovery and clustering
- Regime data splitting and labeling
- Feature engineering and selection
- Advanced matrix operations
- Fractional differentiation
- Regime continuity management
"""

# Import HMM clustering components
from .hmm_clustering import run_enhanced_step

# Import regime and feature engineering components
from .step04_regime_data_splitting import RegimeDataSplittingStep
from .step04_regime_data_splitting_validator import RegimeDataSplittingValidator
from .step04_5_triple_barrier_method_validator import TripleBarrierMethodValidator
from .step05_labeling import LabelingStep
from .step05_labeling_per_regime import PerRegimeLabelingStep
from .step05_labeling_validator import LabelingValidator
from .step06_feature_engineering import FeatureEngineeringStep
from .step06_feature_engineering_per_regime import PerRegimeFeatureEngineeringStep
from .step06_feature_engineering_validator import FeatureEngineeringValidator
from .step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
from .step07_enhanced_matrix_operations_per_regime import PerRegimeMatrixOperationsStep
from .step07_enhanced_matrix_operations_validator import MatrixOperationsValidator
from .step08_advanced_feature_selection import AdvancedFeatureSelectionStep
from .step08_advanced_feature_selection_per_regime import PerRegimeFeatureSelectionStep

# Import additional components
from .vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering
from .vectorized_labelling_orchestrator import VectorizedLabellingOrchestrator
from .fractional_differentiation import FractionalDifferentiation
from .fractional_feature_selector import FractionalFeatureSelector
from .hmm_feature_enhancer import HMMFeatureEnhancer
from .precompute_wavelet_features import PrecomputeWaveletFeatures
from .regime_continuity_decorator import RegimeContinuityDecorator
from .regime_continuity_manager import RegimeContinuityManager
from .regime_continuity_validator import RegimeContinuityValidator
from .regime_handler import RegimeHandler
from .regime_processing_decorator import RegimeProcessingDecorator
from .integrate_regime_processing import IntegrateRegimeProcessing

# Import enhanced orchestrator
from .enhanced_market_analysis_orchestrator import (
    MarketAnalysisPipelineOrchestrator,
    run_enhanced_market_analysis_pipeline,
)

# Main pipeline function - now uses enhanced orchestrator
async def run_market_analysis_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete market analysis pipeline with enhanced validation and error handling."""
    try:
        # Use the enhanced orchestrator for comprehensive pipeline execution
        orchestrator = MarketAnalysisPipelineOrchestrator(config)
        return await orchestrator.execute_pipeline(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            **config
        )
        
    except Exception as e:
        print(f"Market analysis pipeline failed: {e}")
        return False

__all__ = [
    'run_enhanced_step',
    'RegimeDataSplittingStep',
    'RegimeDataSplittingValidator',
    'TripleBarrierMethodValidator',
    'LabelingStep',
    'PerRegimeLabelingStep',
    'LabelingValidator',
    'FeatureEngineeringStep',
    'PerRegimeFeatureEngineeringStep',
    'FeatureEngineeringValidator',
    'EnhancedMatrixOperationsStep',
    'PerRegimeMatrixOperationsStep',
    'MatrixOperationsValidator',
    'AdvancedFeatureSelectionStep',
    'PerRegimeFeatureSelectionStep',
    'VectorizedAdvancedFeatureEngineering',
    'VectorizedLabellingOrchestrator',
    'FractionalDifferentiation',
    'FractionalFeatureSelector',
    'HMMFeatureEnhancer',
    'PrecomputeWaveletFeatures',
    'RegimeContinuityDecorator',
    'RegimeContinuityManager',
    'RegimeContinuityValidator',
    'RegimeHandler',
    'RegimeProcessingDecorator',
    'IntegrateRegimeProcessing',
    'MarketAnalysisPipelineOrchestrator',
    'run_enhanced_market_analysis_pipeline',
    'run_market_analysis_pipeline'
]