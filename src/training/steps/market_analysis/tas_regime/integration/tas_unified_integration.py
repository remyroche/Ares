"""
TAS Unified Integration

This module demonstrates how to integrate the TAS system with the enhanced
unified utilities for economic significance, trading viability, and regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
import time
from datetime import datetime

# Import TAS components
from ..core.tas_engine import TASEngine, TASEngineConfig
from ..core.tas_result import TASResult

# Import enhanced unified utilities
from ...hybrid_nas_tas_regime.shared_utils import (
    UnifiedEconomicSignificanceEvaluator, EconomicEvaluationConfig,
    UnifiedTradingViabilityEvaluator, TradingViabilityConfig,
    UnifiedMultiObjectiveOptimizer, OptimizationConfig,
    UnifiedRegimeAnalyzer, RegimeAnalysisConfig,
    create_unified_economic_evaluator,
    create_unified_trading_viability_evaluator,
    create_unified_multi_objective_optimizer,
    create_unified_regime_analyzer
)

logger = logging.getLogger(__name__)


@dataclass
class TASUnifiedConfig:
    """Configuration for TAS unified integration."""
    
    # TAS configuration
    tas_config: TASEngineConfig = field(default_factory=TASEngineConfig)
    
    # Unified utilities configuration
    economic_config: EconomicEvaluationConfig = field(default_factory=EconomicEvaluationConfig)
    trading_config: TradingViabilityConfig = field(default_factory=TradingViabilityConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    regime_config: RegimeAnalysisConfig = field(default_factory=RegimeAnalysisConfig)
    
    # Integration settings
    enable_economic_evaluation: bool = True
    enable_trading_viability: bool = True
    enable_multi_objective_optimization: bool = True
    enable_regime_analysis: bool = True
    
    # TAS-specific enhancements
    enable_tree_based_analysis: bool = True
    tree_metadata_extraction: bool = True


class TASUnifiedIntegration:
    """
    TAS Unified Integration.
    
    Integrates TAS system with enhanced unified utilities for comprehensive
    regime detection and analysis.
    """
    
    def __init__(self, config: TASUnifiedConfig):
        """Initialize TAS unified integration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize TAS engine
        self.tas_engine = TASEngine(config.tas_config)
        
        # Initialize unified utilities
        self.economic_evaluator = None
        self.trading_evaluator = None
        self.optimizer = None
        self.regime_analyzer = None
        
        if config.enable_economic_evaluation:
            self.economic_evaluator = create_unified_economic_evaluator(config.economic_config)
        
        if config.enable_trading_viability:
            self.trading_evaluator = create_unified_trading_viability_evaluator(config.trading_config)
        
        if config.enable_multi_objective_optimization:
            self.optimizer = create_unified_multi_objective_optimizer(config.optimization_config)
        
        if config.enable_regime_analysis:
            self.regime_analyzer = create_unified_regime_analyzer(config.regime_config)
        
        self.logger.info("✅ TAS Unified Integration initialized")
        self.logger.info(f"   Economic evaluation: {config.enable_economic_evaluation}")
        self.logger.info(f"   Trading viability: {config.enable_trading_viability}")
        self.logger.info(f"   Multi-objective optimization: {config.enable_multi_objective_optimization}")
        self.logger.info(f"   Regime analysis: {config.enable_regime_analysis}")
    
    def search_and_evaluate(self, 
                           market_data: Union[pd.DataFrame, np.ndarray],
                           search_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform TAS search and comprehensive evaluation.
        
        Args:
            market_data: Market data (OHLCV)
            search_config: Optional search configuration
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            self.logger.info("🚀 Starting TAS search and evaluation...")
            start_time = time.time()
            
            # Perform TAS search
            tas_result = self.tas_engine.search(market_data, search_config)
            
            if not tas_result.success:
                self.logger.error("❌ TAS search failed")
                return {'success': False, 'error': 'TAS search failed'}
            
            # Extract regime predictions and metadata
            regime_predictions = tas_result.regime_predictions
            regime_probabilities = tas_result.regime_probabilities
            model_metadata = self._extract_tas_metadata(tas_result)
            
            # Comprehensive evaluation
            evaluation_results = {}
            
            # Economic significance evaluation
            if self.economic_evaluator:
                economic_result = self.economic_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="TAS", model_metadata=model_metadata
                )
                evaluation_results['economic_significance'] = economic_result
            
            # Trading viability evaluation
            if self.trading_evaluator:
                trading_result = self.trading_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="TAS", model_metadata=model_metadata
                )
                evaluation_results['trading_viability'] = trading_result
            
            # Multi-objective optimization
            if self.optimizer:
                optimization_result = self.optimizer.optimize(
                    market_data, regime_predictions, regime_probabilities
                )
                evaluation_results['multi_objective_optimization'] = optimization_result
            
            # Regime analysis
            if self.regime_analyzer:
                regime_analysis = self.regime_analyzer.analyze(
                    regime_predictions, regime_probabilities, market_data,
                    architecture_type="TAS", model_metadata=model_metadata
                )
                evaluation_results['regime_analysis'] = regime_analysis
            
            execution_time = time.time() - start_time
            
            # Compile comprehensive results
            results = {
                'success': True,
                'execution_time': execution_time,
                'tas_result': tas_result,
                'evaluation_results': evaluation_results,
                'model_metadata': model_metadata,
                'architecture_type': 'TAS'
            }
            
            self.logger.info(f"✅ TAS search and evaluation completed in {execution_time:.2f}s")
            self.logger.info(f"   Economic significance: {evaluation_results.get('economic_significance', {}).get('overall_score', 0.0):.3f}")
            self.logger.info(f"   Trading viability: {evaluation_results.get('trading_viability', {}).get('overall_score', 0.0):.3f}")
            self.logger.info(f"   Regime stability: {evaluation_results.get('regime_analysis', {}).get('overall_stability', 0.0):.3f}")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ TAS search and evaluation failed: {e}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    def _extract_tas_metadata(self, tas_result: TASResult) -> Dict[str, Any]:
        """Extract TAS-specific metadata for unified utilities."""
        try:
            metadata = {
                'architecture_type': 'TAS',
                'tree_depth': getattr(tas_result, 'tree_depth', 5),
                'n_leaves': getattr(tas_result, 'n_leaves', 10),
                'complexity': getattr(tas_result, 'complexity', 1.0),
                'interpretability': getattr(tas_result, 'interpretability', 0.8),
                'decision_threshold': getattr(tas_result, 'decision_threshold', 0.6),
                'feature_importance': getattr(tas_result, 'feature_importance', {}),
                'confidence': getattr(tas_result, 'confidence', 0.8),
                'uncertainty': getattr(tas_result, 'uncertainty', None)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"TAS metadata extraction failed: {e}")
            return {'architecture_type': 'TAS'}
    
    def evaluate_regime_quality(self, 
                               market_data: Union[pd.DataFrame, np.ndarray],
                               regime_predictions: np.ndarray,
                               regime_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate regime quality using unified utilities.
        
        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Regime predictions
            regime_probabilities: Optional regime probabilities
            
        Returns:
            Regime quality evaluation results
        """
        try:
            self.logger.info("🔍 Evaluating regime quality...")
            
            # Extract TAS metadata
            model_metadata = {
                'architecture_type': 'TAS',
                'tree_depth': 5,  # Default values
                'n_leaves': 10,
                'complexity': 1.0,
                'interpretability': 0.8,
                'decision_threshold': 0.6,
                'feature_importance': {},
                'confidence': 0.8
            }
            
            evaluation_results = {}
            
            # Economic significance evaluation
            if self.economic_evaluator:
                economic_result = self.economic_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="TAS", model_metadata=model_metadata
                )
                evaluation_results['economic_significance'] = economic_result
            
            # Trading viability evaluation
            if self.trading_evaluator:
                trading_result = self.trading_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="TAS", model_metadata=model_metadata
                )
                evaluation_results['trading_viability'] = trading_result
            
            # Regime analysis
            if self.regime_analyzer:
                regime_analysis = self.regime_analyzer.analyze(
                    regime_predictions, regime_probabilities, market_data,
                    architecture_type="TAS", model_metadata=model_metadata
                )
                evaluation_results['regime_analysis'] = regime_analysis
            
            self.logger.info("✅ Regime quality evaluation completed")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Regime quality evaluation failed: {e}")
            return {'error': str(e)}
    
    def optimize_architecture(self, 
                            market_data: Union[pd.DataFrame, np.ndarray],
                            regime_predictions: np.ndarray,
                            regime_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize TAS architecture using multi-objective optimization.
        
        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Current regime predictions
            regime_probabilities: Optional regime probabilities
            
        Returns:
            Optimization results
        """
        try:
            self.logger.info("🔧 Optimizing TAS architecture...")
            
            if not self.optimizer:
                self.logger.warning("Multi-objective optimizer not available")
                return {'error': 'Multi-objective optimizer not available'}
            
            # Perform multi-objective optimization
            optimization_result = self.optimizer.optimize(
                market_data, regime_predictions, regime_probabilities
            )
            
            self.logger.info("✅ Architecture optimization completed")
            
            return {
                'success': optimization_result.success,
                'optimization_result': optimization_result,
                'best_solution': optimization_result.best_solution,
                'pareto_solutions': optimization_result.pareto_solutions
            }
            
        except Exception as e:
            self.logger.error(f"❌ Architecture optimization failed: {e}")
            return {'error': str(e)}


# Convenience functions
def create_tas_unified_integration(config: Optional[TASUnifiedConfig] = None) -> TASUnifiedIntegration:
    """Create TAS unified integration."""
    if config is None:
        config = TASUnifiedConfig()
    return TASUnifiedIntegration(config)


def quick_tas_evaluation(market_data: Union[pd.DataFrame, np.ndarray],
                        search_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Quick TAS evaluation with default settings."""
    integration = create_tas_unified_integration()
    return integration.search_and_evaluate(market_data, search_config)