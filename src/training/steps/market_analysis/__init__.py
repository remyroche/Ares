"""Market Analysis Module: Contains SR optimization and related market analysis components."""

from .sr_detection import SRDetectionStep
from .sr_clustering import SRClusteringStep
from .sub_pipeline import MarketAnalysisSubPipeline

# Backward compatibility alias
SROptimizationStep = MarketAnalysisSubPipeline

__all__ = [
    'SRDetectionStep', 
    'SRClusteringStep',
    'MarketAnalysisSubPipeline',
    'SROptimizationStep'  # Backward compatibility alias
]