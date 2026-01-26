"""
SR Clustering Module

This module provides backtesting-enhanced clustering for Support/Resistance levels.
"""

from typing import TYPE_CHECKING
from ..logger import system_logger
from src.utils.initialization_guard import init_guard
from src.utils.lazy_module_loader import make_lazy_getattr, make_lazy_dir

# Initialize module logger
logger = system_logger.getChild('sr_clustering')

if init_guard.mark_initialized("utils.sr_clustering"):
    logger.debug("Initializing SR Clustering module")

# Define mapping of export names to submodule names
_EXPORT_MAP = {
    'SRBacktestingEngine': '.sr_backtesting_engine',
    'BacktestConfig': '.sr_backtesting_engine',
    'SRLevel': '.sr_backtesting_engine',
    'BacktestResult': '.sr_backtesting_engine',
    'get_backtesting_engine': '.sr_backtesting_engine',
    
    'BacktestingEnhancedClustering': '.backtesting_enhanced_clustering',
    'BacktestingEnhancedConfig': '.backtesting_enhanced_clustering',
    'get_backtesting_enhanced_clustering': '.backtesting_enhanced_clustering',
    
    'WeightOptimizationEngine': '.weight_optimization_engine',
    'WeightOptimizationConfig': '.weight_optimization_engine',
    'get_weight_optimization_engine': '.weight_optimization_engine',
    
    'PredictiveSREngine': '.predictive_sr_engine',
    'PredictiveConfig': '.predictive_sr_engine',
    'SRPrediction': '.predictive_sr_engine',
    'get_predictive_sr_engine': '.predictive_sr_engine',
    
    'TradingMLIntegration': '.trading_ml_integration',
    'TradingMLConfig': '.trading_ml_integration',
    'TradingSignal': '.trading_ml_integration',
    'get_trading_ml_integration': '.trading_ml_integration'
}

__all__ = list(_EXPORT_MAP.keys())

# Static typing support
if TYPE_CHECKING:
    from .sr_backtesting_engine import SRBacktestingEngine, BacktestConfig, SRLevel, BacktestResult, get_backtesting_engine
    from .backtesting_enhanced_clustering import BacktestingEnhancedClustering, BacktestingEnhancedConfig, get_backtesting_enhanced_clustering
    from .weight_optimization_engine import WeightOptimizationEngine, WeightOptimizationConfig, get_weight_optimization_engine
    from .predictive_sr_engine import PredictiveSREngine, PredictiveConfig, SRPrediction, get_predictive_sr_engine
    from .trading_ml_integration import TradingMLIntegration, TradingMLConfig, TradingSignal, get_trading_ml_integration

# Use generalized lazy loading helpers
__getattr__ = make_lazy_getattr(_EXPORT_MAP, __package__, logger)
__dir__ = make_lazy_dir(_EXPORT_MAP, globals())

if init_guard.is_initialized("utils.sr_clustering"):
    logger.debug("🎉 SR Clustering module initialization completed successfully")
