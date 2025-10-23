"""
Candle-Based Features Research Module

This module contains advanced research implementations for candle pattern analysis
with machine learning-based trading indicators, comprehensive model comparison,
interpretability, and consensus-based indicator selection.

Key Features:
- Series of candles (consecutive patterns, sequences)
- Cross-timeframe candle analysis
- Multi-dimensional interactions (volume, momentum, volatility)
- Pattern strength and quality assessment
- Multiple ML model support (LGBM, Random Forest, GRU, TFT)
- Model comparison and interpretability analysis
- Consensus-based indicator selection
- Comprehensive feature engineering
- Real-time indicator generation
- Performance evaluation and backtesting

Modules:
- ml_candle_pattern_indicators: Core ML indicator generators
- ml_indicator_training_pipeline: Training pipeline with feature engineering
- ml_neural_indicators: Neural network implementations
- ml_indicator_examples: Usage examples and integration patterns
- ml_indicator_integration: Main integration system
- model_comparison_pipeline: Model comparison and consensus system
- interpretability_analysis: Model interpretability and explainability
- consensus_indicator_system: Consensus-based indicator selection
- enhanced_consensus_system: Enhanced consensus with advanced features
- advanced_candle_features: Advanced candle feature engineering
- comprehensive_example: Complete demonstration and analysis
"""

from .ml_candle_pattern_indicators import (
    MLIndicatorGenerator, IndicatorType, ModelType, IndicatorConfig,
    create_ml_indicator_generator
)
from .ml_indicator_training_pipeline import (
    MLIndicatorTrainingPipeline, TrainingConfig, create_training_pipeline
)
from .ml_neural_indicators import (
    NeuralIndicatorGenerator, NeuralConfig, create_neural_indicator_generator
)
from .ml_indicator_integration import (
    MLIndicatorSystem, create_ml_indicator_system
)
from .model_comparison_pipeline import (
    ModelComparisonPipeline, ConsensusConfig, create_model_comparison_pipeline
)
from .interpretability_analysis import (
    InterpretabilityAnalyzer, create_interpretability_analyzer
)
from .consensus_indicator_system import (
    ConsensusIndicatorSystem, create_consensus_system
)
from .enhanced_consensus_system import (
    EnhancedConsensusSystem, EnhancedConsensusConfig, create_enhanced_consensus_system
)
from .advanced_candle_features import (
    AdvancedCandleFeatureGenerator, AdvancedFeatureConfig, create_advanced_candle_feature_generator
)

__all__ = [
    # Core generators
    'MLIndicatorGenerator',
    'NeuralIndicatorGenerator',
    'IndicatorType',
    'ModelType',
    'IndicatorConfig',
    'NeuralConfig',
    
    # Training pipeline
    'MLIndicatorTrainingPipeline',
    'TrainingConfig',
    'create_training_pipeline',
    
    # Model comparison and consensus
    'ModelComparisonPipeline',
    'ConsensusConfig',
    'create_model_comparison_pipeline',
    
    # Interpretability
    'InterpretabilityAnalyzer',
    'create_interpretability_analyzer',
    
    # Consensus system
    'ConsensusIndicatorSystem',
    'create_consensus_system',
    
    # Enhanced consensus system
    'EnhancedConsensusSystem',
    'EnhancedConsensusConfig',
    'create_enhanced_consensus_system',
    
    # Advanced candle features
    'AdvancedCandleFeatureGenerator',
    'AdvancedFeatureConfig',
    'create_advanced_candle_feature_generator',
    
    # Integration
    'MLIndicatorSystem',
    'create_ml_indicator_system',
    'create_ml_indicator_generator',
    'create_neural_indicator_generator'
]