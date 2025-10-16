"""
TAS Integration for Data-Driven Model Selection

This module integrates the data-driven model selector with the TAS regime detection system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime

from .data_driven_model_selector import DataDrivenModelSelector, ModelSelectorConfig
from ..tas_regime.core.tas_regime_detector import TASRegimeDetector
from ..tas_regime.core.tas_regime_config import TASRegimeConfig

logger = logging.getLogger(__name__)

class TASModelSelector:
    """
    TAS-specific model selector that integrates with the TAS Regime Detection System.
    """

    def __init__(self,
                 tas_config: TASRegimeConfig,
                 selector_config: Optional[ModelSelectorConfig] = None):
        """Initialize TAS model selector."""
        self.tas_config = tas_config
        self.selector_config = selector_config or ModelSelectorConfig()

        # Initialize components
        self.tas_detector = TASRegimeDetector(tas_config)
        self.model_selector = DataDrivenModelSelector(self.selector_config)

        # TAS-specific model registry
        self.tas_models = {
            'tree_based_clustering': 'Tree-Based Clustering',
            'statistical_validation': 'Statistical Validation',
            'clvsa_enhanced': 'CLVSA Enhanced Detection',
            'hybrid_detection': 'Hybrid Detection Method',
            'meta_learning_adapted': 'Meta-Learning Adapted',
            'bootstrap_validated': 'Bootstrap Validated',
            'multi_timeframe': 'Multi-Timeframe Detection',
            'real_time_streaming': 'Real-Time Streaming Detection'
        }

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("✅ TAS Model Selector initialized")

    def detect_regimes_and_select_models(self,
                                       market_data: Union[pd.DataFrame, np.ndarray],
                                       timestamps: Optional[np.ndarray] = None,
                                       available_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect regimes and select optimal models for each regime.

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            available_models: List of available TAS models

        Returns:
            Dictionary with regime detection and model selection results
        """
        try:
            start_time = time.time()

            # Step 1: Detect regimes using TAS
            self.logger.info("🌲 Detecting regimes using TAS system...")
            tas_result = self.tas_detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                optimize_performance=True,
                enable_patchtst_enhancement=True
            )

            if not tas_result.success:
                raise ValueError(f"TAS regime detection failed: {tas_result.error_message}")

            # Step 2: Get available models
            if available_models is None:
                available_models = list(self.tas_models.keys())

            # Step 3: Select models for each detected regime
            regime_model_selections = {}
            unique_regimes = np.unique(tas_result.regime_predictions)

            for regime_id in unique_regimes:
                self.logger.info(f"🎯 Selecting models for regime {regime_id}...")

                # Get regime-specific data
                regime_mask = tas_result.regime_predictions == regime_id
                regime_data = market_data[regime_mask] if hasattr(market_data, '__getitem__') else market_data[regime_mask]

                # Select best model for this regime
                selected_model, ensemble_weights = self.model_selector.select_model_for_regime(
                    regime_id=int(regime_id),
                    available_models=available_models
                )

                # Get ensemble weights if enabled
                if self.selector_config.enable_ensemble:
                    ensemble_weights = self.model_selector.get_ensemble_weights(
                        regime_id=int(regime_id),
                        available_models=available_models
                    )

                regime_model_selections[regime_id] = {
                    'selected_model': selected_model,
                    'ensemble_weights': ensemble_weights,
                    'regime_characteristics': self._extract_regime_characteristics(
                        regime_data, tas_result, regime_id
                    ),
                    'regime_confidence': tas_result.regime_stability_scores[regime_mask].mean() if len(tas_result.regime_stability_scores) > 0 else 0.5
                }

            execution_time = time.time() - start_time

            # Create comprehensive result
            result = {
                'success': True,
                'execution_time': execution_time,
                'tas_result': tas_result,
                'regime_model_selections': regime_model_selections,
                'system_summary': self.model_selector.get_system_summary(),
                'metadata': {
                    'system': 'TAS Data-Driven Model Selection',
                    'n_regimes_detected': len(unique_regimes),
                    'models_available': len(available_models),
                    'ensemble_enabled': self.selector_config.enable_ensemble,
                    'timestamp': datetime.now().isoformat()
                }
            }

            self.logger.info(f"✅ TAS regime detection and model selection completed in {execution_time:.2f}s")
            self.logger.info(f"   Regimes detected: {len(unique_regimes)}")
            self.logger.info(f"   Models available: {len(available_models)}")

            return result

        except Exception as e:
            self.logger.error(f"❌ TAS regime detection and model selection failed: {e}")
            return {
                'success': False,
                'error_message': str(e),
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0.0
            }

    def update_model_performance(self,
                               regime_id: int,
                               model_name: str,
                               predictions: np.ndarray,
                               actual_values: np.ndarray,
                               execution_time: float,
                               regime_characteristics: Optional[Dict[str, Any]] = None):
        """
        Update model performance for a specific regime.

        Args:
            regime_id: ID of the market regime
            model_name: Name of the TAS model
            predictions: Model predictions
            actual_values: Actual values
            execution_time: Time taken for inference
            regime_characteristics: Characteristics of the regime
        """
        try:
            # Update performance in the model selector
            metrics = self.model_selector.register_model_performance(
                regime_id=regime_id,
                model_name=model_name,
                predictions=predictions,
                actual_values=actual_values,
                execution_time=execution_time,
                regime_characteristics=regime_characteristics
            )

            self.logger.info(f"Updated performance for TAS model {model_name} in regime {regime_id}: "
                           f"F1={metrics.f1_score:.3f}, Accuracy={metrics.accuracy:.3f}")

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to update TAS model performance: {e}")
            raise

    def get_regime_insights(self, regime_id: int) -> Dict[str, Any]:
        """Get insights about model performance in a specific regime."""
        return self.model_selector.get_regime_insights(regime_id)

    def get_optimal_model_for_regime(self, regime_id: int) -> Tuple[str, Dict[str, float]]:
        """Get the optimal model for a specific regime."""
        available_models = list(self.tas_models.keys())
        return self.model_selector.select_model_for_regime(regime_id, available_models)

    def _extract_regime_characteristics(self,
                                      regime_data: np.ndarray,
                                      tas_result,
                                      regime_id: int) -> Dict[str, Any]:
        """Extract characteristics of a regime for model selection."""
        try:
            characteristics = {
                'regime_id': regime_id,
                'data_size': len(regime_data),
                'volatility': np.std(regime_data) if len(regime_data) > 0 else 0.0,
                'mean_value': np.mean(regime_data) if len(regime_data) > 0 else 0.0,
                'trend_strength': 0.0,  # Would need to calculate trend
                'complexity_score': 0.0,  # Would need to calculate complexity
                'tas_confidence': 0.0
            }

            # Add TAS-specific characteristics
            if hasattr(tas_result, 'economic_significance_scores'):
                regime_mask = tas_result.regime_predictions == regime_id
                if len(tas_result.economic_significance_scores) > 0:
                    characteristics['economic_significance'] = np.mean(tas_result.economic_significance_scores[regime_mask])

            if hasattr(tas_result, 'trading_viability_scores'):
                regime_mask = tas_result.regime_predictions == regime_id
                if len(tas_result.trading_viability_scores) > 0:
                    characteristics['trading_viability'] = np.mean(tas_result.trading_viability_scores[regime_mask])

            if hasattr(tas_result, 'regime_stability_scores'):
                regime_mask = tas_result.regime_predictions == regime_id
                if len(tas_result.regime_stability_scores) > 0:
                    characteristics['stability_score'] = np.mean(tas_result.regime_stability_scores[regime_mask])

            # Add TAS-specific tree performance metrics
            if hasattr(tas_result, 'tree_performance_metrics') and tas_result.tree_performance_metrics:
                characteristics['tree_performance'] = tas_result.tree_performance_metrics

            # Add CLVSA enhanced features if available
            if hasattr(tas_result, 'clvsa_enhanced_features') and tas_result.clvsa_enhanced_features is not None:
                regime_mask = tas_result.regime_predictions == regime_id
                if len(tas_result.clvsa_enhanced_features) > 0:
                    characteristics['clvsa_features'] = np.mean(tas_result.clvsa_enhanced_features[regime_mask])

            return characteristics

        except Exception as e:
            self.logger.error(f"Failed to extract regime characteristics: {e}")
            return {'regime_id': regime_id, 'error': str(e)}

    def save_mappings(self):
        """Save current model mappings."""
        self.model_selector.save_mappings()

    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the TAS model selection system."""
        summary = self.model_selector.get_system_summary()
        summary['system_type'] = 'TAS Data-Driven Model Selection'
        summary['tas_models_available'] = len(self.tas_models)
        summary['tas_config'] = {
            'primary_architecture': self.tas_config.primary_architecture.value,
            'enable_statistical_methods': self.tas_config.enable_statistical_methods,
            'enable_economic_evaluation': self.tas_config.enable_economic_evaluation,
            'enable_meta_learning': self.tas_config.enable_meta_learning
        }
        return summary
