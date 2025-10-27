"""
SR Clustering Module

This module provides backtesting-enhanced clustering for Support/Resistance levels.
"""

import logging
from ..logger import system_logger

# Initialize module logger
logger = system_logger.getChild('sr_clustering')

logger.info("Initializing SR Clustering module")

try:
    from .sr_backtesting_engine import SRBacktestingEngine, BacktestConfig, SRLevel, BacktestResult, get_backtesting_engine
    logger.info("✅ Successfully imported SRBacktestingEngine components")
except ImportError as e:
    logger.error(f"❌ Failed to import SRBacktestingEngine components: {e}")
    raise

try:
    from .backtesting_enhanced_clustering import BacktestingEnhancedClustering, BacktestingEnhancedConfig, get_backtesting_enhanced_clustering
    logger.info("✅ Successfully imported BacktestingEnhancedClustering components")
except ImportError as e:
    logger.error(f"❌ Failed to import BacktestingEnhancedClustering components: {e}")
    raise

try:
    from .weight_optimization_engine import WeightOptimizationEngine, WeightOptimizationConfig, get_weight_optimization_engine
    logger.info("✅ Successfully imported WeightOptimizationEngine components")
except ImportError as e:
    logger.error(f"❌ Failed to import WeightOptimizationEngine components: {e}")
    raise

try:
    from .predictive_sr_engine import PredictiveSREngine, PredictiveConfig, SRPrediction, get_predictive_sr_engine
    logger.info("✅ Successfully imported PredictiveSREngine components")
except ImportError as e:
    logger.error(f"❌ Failed to import PredictiveSREngine components: {e}")
    raise

try:
    from .trading_ml_integration import TradingMLIntegration, TradingMLConfig, TradingSignal, get_trading_ml_integration
    logger.info("✅ Successfully imported TradingMLIntegration components")
except ImportError as e:
    logger.error(f"❌ Failed to import TradingMLIntegration components: {e}")
    raise

logger.info("🎉 SR Clustering module initialization completed successfully")

__all__ = [
    'SRBacktestingEngine',
    'BacktestConfig',
    'SRLevel',
    'BacktestResult',
    'get_backtesting_engine',
    'BacktestingEnhancedClustering',
    'BacktestingEnhancedConfig',
    'get_backtesting_enhanced_clustering',
    'WeightOptimizationEngine',
    'WeightOptimizationConfig',
    'get_weight_optimization_engine',
    'PredictiveSREngine',
    'PredictiveConfig',
    'SRPrediction',
    'get_predictive_sr_engine',
    'TradingMLIntegration',
    'TradingMLConfig',
    'TradingSignal',
    'get_trading_ml_integration'
]

# Enhanced SR Clustering components are now integrated into the main SRClusteringComponent
