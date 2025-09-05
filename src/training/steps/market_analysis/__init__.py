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

# Import regime and feature engineering components
# from .step04_5_triple_barrier_method_validator import TripleBarrierMethodValidator
# from .step05_labeling_per_regime import PerRegimeLabelingStep
# from .step06_feature_engineering import FeatureEngineeringStep
# from .step06_feature_engineering_per_regime import PerRegimeFeatureEngineeringStep

# Import additional components

# Import enhanced orchestrator
# from .enhanced_market_analysis_orchestrator import (
#     MarketAnalysisPipelineOrchestrator,
#     run_enhanced_market_analysis_pipeline,
# )

# Import enhanced logging system

# Main pipeline function - now uses enhanced orchestrator
async def run_market_analysis_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete market analysis pipeline with enhanced validation and error handling."""
    try:
        # Use the enhanced orchestrator for comprehensive pipeline execution
        from .enhanced_market_analysis_orchestrator import MarketAnalysisPipelineOrchestrator
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
    'run_market_analysis_pipeline',
    'EnhancedPipelineLogger',
    'enhanced_logger',
    'ProgressMonitor',
    'progress_monitor',
    'ProgressContext'
]