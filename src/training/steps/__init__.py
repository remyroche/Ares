"""Training steps module with explicit imports.

This module provides access to all training step classes and functions
with explicit imports to avoid namespace pollution.
"""

# Import specific classes from each step module
try:
    passpassfrom .step01_data_collection import DataCollectionStep
except ImportError: DataCollectionStep = None

try:
    passfrom .step02_feature_engineering import FeatureEngineeringStep
except ImportError: FeatureEngineeringStep = None

try:
    passfrom .step03_hmm_regime_discovery import HMMRegimeDiscoveryStep
except ImportError: HMMRegimeDiscoveryStep = None

try:
    passfrom .step04_processing_labeling import ProcessingLabelingStep
except ImportError: ProcessingLabelingStep = None

try:
    passfrom .step04_market_regime_classification import MarketRegimeClassificationStep
except ImportError: MarketRegimeClassificationStep = None

try:
    passfrom .step05_regime_data_splitting import RegimeDataSplittingStep
except ImportError: RegimeDataSplittingStep = None

try:
    passfrom .step05_5_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
except ImportError: UnifiedRegimeIntelligenceStep = None

try:
    passfrom .step09_hmm_based_training import HMMBasedTrainingStep
except ImportError: HMMBasedTrainingStep = None

try:
    passfrom .step07_analyst_enhancement import AnalystEnhancementStep
except ImportError: AnalystEnhancementStep = None

try:
    passfrom .step08_tactician_labeling import TacticianLabelingStep
except ImportError: TacticianLabelingStep = None

try:
    passfrom .step09_tactician_specialist_training import TacticianSpecialistTrainingStep
except ImportError: TacticianSpecialistTrainingStep = None

try:
    passfrom .step10_confidence_calibration import ConfidenceCalibrationStep
except ImportError: ConfidenceCalibrationStep = None

try:
    passfrom .step11_final_parameters_optimization import FinalParametersOptimizationStep
except ImportError: FinalParametersOptimizationStep = None

try:
    passfrom .step12_walk_forward_validation import WalkForwardValidationStep
except ImportError: WalkForwardValidationStep = None

try:
    passfrom .step13_monte_carlo_validation import MonteCarloValidationStep
except ImportError: MonteCarloValidationStep = None

try:
    passfrom .step14_ab_testing import ABTestingStep
except ImportError: ABTestingStep = None

try:
    passfrom .step15_saving import SavingStep
except ImportError: SavingStep = None

# Import utility functions
try:
    passfrom .unified_data_loader import UnifiedDataLoader, get_unified_data_loader
except ImportError: get_unified_data_loader = None
    UnifiedDataLoader = None
# Export all available classes
__all__, [
    "ABTestingStep",
    "AnalystEnhancementStep",
    "ConfidenceCalibrationStep",
    "DataCollectionStep",
    "FeatureEngineeringStep",
    "FinalParametersOptimizationStep",
    "HMMBasedTrainingStep",
    "HMMRegimeDiscoveryStep",
    "MarketRegimeClassificationStep",
    "MonteCarloValidationStep",
    "ProcessingLabelingStep",
    "RegimeDataSplittingStep",
    "SavingStep",
    "TacticianLabelingStep",
    "TacticianSpecialistTrainingStep",
    "UnifiedDataLoader",
    "UnifiedRegimeIntelligenceStep",
    "WalkForwardValidationStep",
    "get_unified_data_loader",
]