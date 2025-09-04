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

# Main pipeline function
async def run_market_analysis_pipeline(symbol, exchange, timeframe, data_dir, **config):
    """Run the complete market analysis pipeline."""
    try:
        # Step 1: HMM Clustering (if enabled)
        if config.get('hmm_clustering', True):
            await run_enhanced_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=config.get('force_rerun', True)
            )
        
        # Step 2: Regime Data Splitting (if enabled)
        if config.get('regime_splitting', True):
            regime_splitter = RegimeDataSplittingStep()
            await regime_splitter.split_regime_data(symbol, exchange, timeframe, data_dir)
        
        # Step 3: Feature Engineering (if enabled)
        if config.get('feature_engineering', True):
            feature_engineer = FeatureEngineeringStep()
            await feature_engineer.engineer_features(symbol, exchange, timeframe, data_dir)
        
        # Step 4: Matrix Operations (if enabled)
        if config.get('matrix_operations', True):
            matrix_ops = EnhancedMatrixOperationsStep()
            await matrix_ops.perform_matrix_operations(symbol, exchange, timeframe, data_dir)
        
        # Step 5: Feature Selection (if enabled)
        if config.get('feature_selection', True):
            feature_selector = AdvancedFeatureSelectionStep()
            await feature_selector.select_features(symbol, exchange, timeframe, data_dir)
        
        return True
        
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
    'run_market_analysis_pipeline'
]