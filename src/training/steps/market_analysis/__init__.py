"""Market Analysis Module: Contains SR optimization and related market analysis components."""

from .sr_detection import SRDetectionStep
from .sr_clustering import SRClusteringStep
from .sr_ml_learning import SRMLLearningStep
from .sub_pipeline import MarketAnalysisSubPipeline

# For backward compatibility, alias the sub_pipeline as the main orchestrator
SROptimizationStep = MarketAnalysisSubPipeline

__all__ = [
    'SRDetectionStep', 
    'SRClusteringStep',
    'SRMLLearningStep',
    'MarketAnalysisSubPipeline',
    'SROptimizationStep'  # Backward compatibility alias
]