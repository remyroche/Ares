"""
Multi-Tier Trading System

A sophisticated three-tier trading system that combines HMM regime detection, 
Analyst decision making, and Tactician timing prediction for optimal trading execution.

System Architecture:
- HMM (1h base, runs every 15 minutes, 100 features, 15-25 regimes)
- Analyst (5m base, runs every 2 minutes, 300+ features, per-regime training)
- Tactician (1m base, runs every 30 seconds, green light dependent)

Model Configurations:
- HMM: CatBoost, Elastic Net (base) + XGBoost (meta-learner)
- Analyst: TCN, CatBoost, LightGBM (base) + Elastic Net (meta-learner)
- Tactician: XGBoost, Random Forest, CatBoost, Elastic Net (base) + LightGBM (meta-learner)
"""

from .multi_tier_orchestrator import (
    MultiTierTradingOrchestrator,
    HMMOutput,
    AnalystOutput,
    TacticianOutput,
    TradingDecision,
    create_multi_tier_trading_orchestrator
)

from .live_execution_system import (
    LiveExecutionSystem,
    ExecutionStatus,
    ExecutionMetrics,
    create_live_execution_system
)

from .enhanced_model_configs import (
    MultiTierModelConfigs,
    ModelConfig,
    TierModelConfig
)

from .feature_extraction import (
    HMMFeatureExtractor,
    AnalystFeatureExtractor,
    TacticianFeatureExtractor,
    create_hmm_feature_extractor,
    create_analyst_feature_extractor,
    create_tactician_feature_extractor
)

__all__ = [
    # Main orchestrator
    'MultiTierTradingOrchestrator',
    'create_multi_tier_trading_orchestrator',
    
    # Live execution system
    'LiveExecutionSystem',
    'create_live_execution_system',
    'ExecutionStatus',
    'ExecutionMetrics',
    
    # Data structures
    'HMMOutput',
    'AnalystOutput', 
    'TacticianOutput',
    'TradingDecision',
    
    # Model configurations
    'MultiTierModelConfigs',
    'ModelConfig',
    'TierModelConfig',
    
    # Feature extractors
    'HMMFeatureExtractor',
    'AnalystFeatureExtractor',
    'TacticianFeatureExtractor',
    'create_hmm_feature_extractor',
    'create_analyst_feature_extractor',
    'create_tactician_feature_extractor'
]