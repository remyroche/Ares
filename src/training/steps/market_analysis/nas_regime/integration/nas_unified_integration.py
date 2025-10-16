"""
NAS Unified Integration

This module demonstrates how to integrate the NAS system with the enhanced
unified utilities for economic significance, trading viability, and regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
import time
from datetime import datetime
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Import NAS components from centralized utilities
from src.utils.nas_tas.core.nas_engine import NASEngine
from src.utils.nas_tas.optimization.architecture_search import ArchitectureSearchOptimizer, ArchitectureSearchConfig

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
class NASUnifiedConfig:
    """Configuration for NAS unified integration."""

    # NAS configuration
    nas_config: ArchitectureSearchConfig = field(default_factory=ArchitectureSearchConfig)

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

    # NAS-specific enhancements
    enable_neural_based_analysis: bool = True
    neural_metadata_extraction: bool = True

class NASUnifiedIntegration:
    """
    NAS Unified Integration.

    Integrates NAS system with enhanced unified utilities for comprehensive
    regime detection and analysis.
    """

    def __init__(self, config: NASUnifiedConfig):
        """Initialize NAS unified integration."""
        tprint("🚀 [NAS_UNIFIED_INTEGRATION] Initializing NAS Unified Integration", color="cyan", bold=True)
        tprint(f"📊 [NAS_UNIFIED_INTEGRATION] Economic evaluation: {config.enable_economic_evaluation}", color="blue")
        tprint(f"📊 [NAS_UNIFIED_INTEGRATION] Trading viability: {config.enable_trading_viability}", color="blue")
        tprint(f"📊 [NAS_UNIFIED_INTEGRATION] Multi-objective optimization: {config.enable_multi_objective_optimization}", color="blue")
        tprint(f"📊 [NAS_UNIFIED_INTEGRATION] Regime analysis: {config.enable_regime_analysis}", color="blue")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize NAS engine
        tprint("🧠 [NAS_UNIFIED_INTEGRATION] Initializing NAS engine", color="yellow")
        self.nas_engine = NASEngine()
        self.architecture_optimizer = ArchitectureSearchOptimizer(config.nas_config)

        # Initialize unified utilities
        tprint("🔧 [NAS_UNIFIED_INTEGRATION] Initializing unified utilities", color="yellow")
        self.economic_evaluator = None
        self.trading_evaluator = None
        self.optimizer = None
        self.regime_analyzer = None

        if config.enable_economic_evaluation:
            tprint("💰 [NAS_UNIFIED_INTEGRATION] Creating economic evaluator", color="yellow")
            self.economic_evaluator = create_unified_economic_evaluator(config.economic_config)

        if config.enable_trading_viability:
            tprint("📈 [NAS_UNIFIED_INTEGRATION] Creating trading viability evaluator", color="yellow")
            self.trading_evaluator = create_unified_trading_viability_evaluator(config.trading_config)

        if config.enable_multi_objective_optimization:
            tprint("🎯 [NAS_UNIFIED_INTEGRATION] Creating multi-objective optimizer", color="yellow")
            self.optimizer = create_unified_multi_objective_optimizer(config.optimization_config)

        if config.enable_regime_analysis:
            tprint("🔍 [NAS_UNIFIED_INTEGRATION] Creating regime analyzer", color="yellow")
            self.regime_analyzer = create_unified_regime_analyzer(config.regime_config)

        tprint("✅ [NAS_UNIFIED_INTEGRATION] NAS Unified Integration initialized successfully", color="green")
        self.logger.info("✅ NAS Unified Integration initialized")
        self.logger.info(f"   Economic evaluation: {config.enable_economic_evaluation}")
        self.logger.info(f"   Trading viability: {config.enable_trading_viability}")
        self.logger.info(f"   Multi-objective optimization: {config.enable_multi_objective_optimization}")
        self.logger.info(f"   Regime analysis: {config.enable_regime_analysis}")

    def search_and_evaluate(self,
                           market_data: Union[pd.DataFrame, np.ndarray],
                           search_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform NAS search and comprehensive evaluation.

        Args:
            market_data: Market data (OHLCV)
            search_config: Optional search configuration

        Returns:
            Comprehensive evaluation results
        """
        try:
            self.logger.info("🚀 Starting NAS search and evaluation...")
            start_time = time.time()

            # Perform NAS search
            nas_result = self.nas_engine.search(market_data, search_config)

            if not nas_result.success:
                self.logger.error("❌ NAS search failed")
                return {'success': False, 'error': 'NAS search failed'}

            # Extract regime predictions and metadata
            regime_predictions = nas_result.regime_predictions
            regime_probabilities = nas_result.regime_probabilities
            model_metadata = self._extract_nas_metadata(nas_result)

            # Comprehensive evaluation
            evaluation_results = {}

            # Economic significance evaluation
            if self.economic_evaluator:
                economic_result = self.economic_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="NAS", model_metadata=model_metadata
                )
                evaluation_results['economic_significance'] = economic_result

            # Trading viability evaluation
            if self.trading_evaluator:
                trading_result = self.trading_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="NAS", model_metadata=model_metadata
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
                    architecture_type="NAS", model_metadata=model_metadata
                )
                evaluation_results['regime_analysis'] = regime_analysis

            execution_time = time.time() - start_time

            # Compile comprehensive results
            results = {
                'success': True,
                'execution_time': execution_time,
                'nas_result': nas_result,
                'evaluation_results': evaluation_results,
                'model_metadata': model_metadata,
                'architecture_type': 'NAS'
            }

            self.logger.info(f"✅ NAS search and evaluation completed in {execution_time:.2f}s")
            self.logger.info(f"   Economic significance: {evaluation_results.get('economic_significance', {}).get('overall_score', 0.0):.3f}")
            self.logger.info(f"   Trading viability: {evaluation_results.get('trading_viability', {}).get('overall_score', 0.0):.3f}")
            self.logger.info(f"   Regime stability: {evaluation_results.get('regime_analysis', {}).get('overall_stability', 0.0):.3f}")

            return results

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS search and evaluation failed: {e}")

            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }

    def _extract_nas_metadata(self, nas_result: NASResult) -> Dict[str, Any]:
        """Extract NAS-specific metadata for unified utilities."""
        try:
            metadata = {
                'architecture_type': 'NAS',
                'confidence': getattr(nas_result, 'confidence', 0.8),
                'architecture_complexity': getattr(nas_result, 'architecture_complexity', 0.5),
                'architecture_efficiency': getattr(nas_result, 'architecture_efficiency', 0.7),
                'uncertainty': getattr(nas_result, 'uncertainty', None),
                'model_parameters': getattr(nas_result, 'model_parameters', {}),
                'training_metrics': getattr(nas_result, 'training_metrics', {}),
                'validation_metrics': getattr(nas_result, 'validation_metrics', {})
            }

            return metadata

        except Exception as e:
            self.logger.warning(f"NAS metadata extraction failed: {e}")
            return {'architecture_type': 'NAS'}

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

            # Extract NAS metadata
            model_metadata = {
                'architecture_type': 'NAS',
                'confidence': 0.8,  # Default values
                'architecture_complexity': 0.5,
                'architecture_efficiency': 0.7,
                'uncertainty': None,
                'model_parameters': {},
                'training_metrics': {},
                'validation_metrics': {}
            }

            evaluation_results = {}

            # Economic significance evaluation
            if self.economic_evaluator:
                economic_result = self.economic_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="NAS", model_metadata=model_metadata
                )
                evaluation_results['economic_significance'] = economic_result

            # Trading viability evaluation
            if self.trading_evaluator:
                trading_result = self.trading_evaluator.evaluate(
                    market_data, regime_predictions, regime_probabilities,
                    architecture_type="NAS", model_metadata=model_metadata
                )
                evaluation_results['trading_viability'] = trading_result

            # Regime analysis
            if self.regime_analyzer:
                regime_analysis = self.regime_analyzer.analyze(
                    regime_predictions, regime_probabilities, market_data,
                    architecture_type="NAS", model_metadata=model_metadata
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
        Optimize NAS architecture using multi-objective optimization.

        Args:
            market_data: Market data (OHLCV)
            regime_predictions: Current regime predictions
            regime_probabilities: Optional regime probabilities

        Returns:
            Optimization results
        """
        try:
            self.logger.info("🔧 Optimizing NAS architecture...")

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
def create_nas_unified_integration(config: Optional[NASUnifiedConfig] = None) -> NASUnifiedIntegration:
    """Create NAS unified integration."""
    if config is None:
        config = NASUnifiedConfig()
    return NASUnifiedIntegration(config)

def quick_nas_evaluation(market_data: Union[pd.DataFrame, np.ndarray],
                        search_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Quick NAS evaluation with default settings."""
    integration = create_nas_unified_integration()
    return integration.search_and_evaluate(market_data, search_config)
