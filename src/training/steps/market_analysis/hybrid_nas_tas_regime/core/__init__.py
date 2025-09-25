"""
Core components for Hybrid NAS-TAS Regime System
"""

from .hybrid_regime_detector import HybridNASTASRegimeDetector
from .unified_architecture_search_engine import UnifiedArchitectureSearchEngine
from .performance_estimator import UnifiedPerformanceEstimator
# Advanced search strategies removed - now using unified search algorithms from src.utils.nas_tas
from ..shared_utils import UnifiedMultiObjectiveOptimizer
from .architecture_encoder import UnifiedArchitectureEncoder
from .nas_financial_features import NASFinancialFeatureEngineer, FeatureSet
from .nas_financial_optimizer import NASFinancialOptimizer, FinancialLossFunctions
from .architecture_signal_generator import ArchitectureSignalGenerator, TradingSignal

__all__ = [
    'HybridNASTASRegimeDetector',
    'UnifiedArchitectureSearchEngine',
    'UnifiedPerformanceEstimator',
    'AdvancedSearchStrategies',
    'UnifiedMultiObjectiveOptimizer',
    'UnifiedArchitectureEncoder',
    'NASFinancialFeatureEngineer',
    'FeatureSet',
    'NASFinancialOptimizer',
    'FinancialLossFunctions',
    'ArchitectureSignalGenerator',
    'TradingSignal'
]