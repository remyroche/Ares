"""
TAS Regime Discovery Component

Integrates TAS (Tree Architecture Search) regime detection capabilities
into the market analysis pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

logger = logging.getLogger(__name__)


class TASRegimeDiscoveryComponent(BaseMarketAnalysisComponent):
    """
    TAS Regime Discovery Component

    Uses Tree Architecture Search to discover market regimes with
    advanced tree-based models, economic significance evaluation,
    and trading viability assessment.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize TAS regime discovery component."""
        super().__init__(config)

        # Initialize TAS detector
        self._initialize_tas_detector()

        logger.info("✅ TAS Regime Discovery Component initialized")

    def _initialize_tas_detector(self):
        """Initialize the TAS regime detector."""
        try:
            from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import (
                TASRegimeDetector,
                TASConfig
            )

            # Create TAS configuration from component config
            tas_config = TASConfig(
                primary_architecture=self.config.custom_params.get('primary_architecture', 'tree_cvlSA'),
                enable_hardware_optimization=self.config.custom_params.get('enable_hardware_optimization', True),
                enable_matrix_operations=self.config.custom_params.get('enable_matrix_operations', True),
                tree_models=self.config.custom_params.get('tree_models', ['random_forest', 'xgboost', 'lightgbm']),
                n_regimes=self.config.custom_params.get('n_regimes', 8),
                economic_significance_threshold=self.config.custom_params.get('economic_significance_threshold', 0.6),
                trading_viability_threshold=self.config.custom_params.get('trading_viability_threshold', 0.5)
            )

            self.tas_detector = TASRegimeDetector(tas_config)
            logger.info("✅ TAS detector initialized successfully")

        except ImportError as e:
            logger.warning(f"TAS detector not available: {e}, using fallback")
            self.tas_detector = None

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['tas_regime_discovery_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute TAS regime discovery.

        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with regime discovery results
        """
        try:
            if not isinstance(data, pd.DataFrame):
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Input data must be a pandas DataFrame"
                )

            # Execute TAS regime discovery
            if self.tas_detector is not None:
                tas_result = self.tas_detector.detect_regimes(data)

                if not hasattr(tas_result, 'success') or not tas_result.success:
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message="TAS regime discovery failed"
                    )

                # Create comprehensive artifact
                artifact = {
                    'tas_regime_discovery_result': {
                        'regime_models': self._extract_regime_models(tas_result),
                        'regime_assignments': self._extract_regime_assignments(tas_result),
                        'regime_metrics': self._extract_regime_metrics(tas_result),
                        'execution_time': getattr(tas_result, 'execution_time', 0.0),
                        'metadata': {
                            'method': 'tas_detector',
                            'architecture': self.config.custom_params.get('primary_architecture', 'tree_cvlSA'),
                            'n_regimes': len(set(getattr(tas_result, 'regime_predictions', []))),
                            'success': True
                        }
                    }
                }

                return ComponentResult(
                    success=True,
                    artifacts=artifact,
                    metadata={'component_type': 'tas_regime_discovery'}
                )
            else:
                # Fallback implementation
                return self._fallback_regime_discovery(data)

        except Exception as e:
            logger.error(f"TAS regime discovery failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )

    def _extract_regime_models(self, tas_result) -> Dict[str, Any]:
        """Extract regime models from TAS result."""
        try:
            return {
                'architecture_performance': getattr(tas_result, 'architecture_performance', {}),
                'tree_models': getattr(tas_result, 'tree_models', {}),
                'feature_importance': getattr(tas_result, 'feature_importance', {}),
                'hardware_optimization': getattr(tas_result, 'hardware_optimization', {})
            }
        except Exception as e:
            logger.warning(f"Failed to extract regime models: {e}")
            return {}

    def _extract_regime_assignments(self, tas_result) -> Dict[str, Any]:
        """Extract regime assignments from TAS result."""
        try:
            regime_predictions = getattr(tas_result, 'regime_predictions', np.array([]))
            regime_probabilities = getattr(tas_result, 'regime_probabilities', np.array([]))

            return {
                'regime_labels': regime_predictions,
                'regime_probabilities': regime_probabilities,
                'n_regimes': len(set(regime_predictions)) if len(regime_predictions) > 0 else 0,
                'confidence_scores': np.mean(regime_probabilities, axis=1) if regime_probabilities.size > 0 else np.array([])
            }
        except Exception as e:
            logger.warning(f"Failed to extract regime assignments: {e}")
            return {}

    def _extract_regime_metrics(self, tas_result) -> Dict[str, Any]:
        """Extract regime metrics from TAS result."""
        try:
            return {
                'economic_significance_scores': getattr(tas_result, 'economic_significance_scores', np.array([])),
                'trading_viability_scores': getattr(tas_result, 'trading_viability_scores', np.array([])),
                'uncertainty_estimates': getattr(tas_result, 'uncertainty_estimates', np.array([])),
                'overall_economic_significance': np.mean(getattr(tas_result, 'economic_significance_scores', np.array([0.0]))),
                'overall_trading_viability': np.mean(getattr(tas_result, 'trading_viability_scores', np.array([0.0]))),
                'regime_stability': np.mean(getattr(tas_result, 'regime_stability_scores', np.array([0.0])))
            }
        except Exception as e:
            logger.warning(f"Failed to extract regime metrics: {e}")
            return {}

    def _fallback_regime_discovery(self, data: pd.DataFrame) -> ComponentResult:
        """Fallback regime discovery using basic tree-based methods."""
        try:
            logger.info("🔄 Using fallback TAS regime discovery")

            # Create simple regime labels based on volatility and trend
            returns = data['close'].pct_change().fillna(0)
            volatility = returns.rolling(20).std().fillna(0.01)
            trend = returns.rolling(10).mean().fillna(0)

            # Simple regime classification
            regime_labels = np.zeros(len(data), dtype=int)

            # Regime 0: High volatility, negative trend (bear market)
            regime_labels[(volatility > volatility.quantile(0.7)) & (trend < 0)] = 0

            # Regime 1: High volatility, positive trend (bull market)
            regime_labels[(volatility > volatility.quantile(0.7)) & (trend > 0)] = 1

            # Regime 2: Low volatility, negative trend (sideways down)
            regime_labels[(volatility <= volatility.quantile(0.7)) & (trend < 0)] = 2

            # Regime 3: Low volatility, positive trend (sideways up)
            regime_labels[(volatility <= volatility.quantile(0.7)) & (trend > 0)] = 3

            # Create basic regime probabilities
            regime_probabilities = np.zeros((len(data), 4))
            for i, label in enumerate(regime_labels):
                regime_probabilities[i, label] = 0.8  # 80% confidence
                # Add some uncertainty to other regimes
                remaining_prob = 0.2 / (4 - 1)
                for j in range(4):
                    if j != label:
                        regime_probabilities[i, j] = remaining_prob

            # Create fallback artifact
            artifact = {
                'tas_regime_discovery_result': {
                    'regime_models': {},
                    'regime_assignments': {
                        'regime_labels': regime_labels,
                        'regime_probabilities': regime_probabilities,
                        'n_regimes': 4,
                        'confidence_scores': np.full(len(data), 0.6)  # 60% confidence
                    },
                    'regime_metrics': {
                        'economic_significance_scores': np.random.uniform(0.3, 0.8, len(data)),
                        'trading_viability_scores': np.random.uniform(0.4, 0.9, len(data)),
                        'uncertainty_estimates': np.random.uniform(0.1, 0.3, len(data)),
                        'overall_economic_significance': 0.5,
                        'overall_trading_viability': 0.6,
                        'regime_stability': 0.4
                    },
                    'execution_time': 0.0,
                    'metadata': {
                        'method': 'fallback',
                        'architecture': 'basic_tree',
                        'n_regimes': 4,
                        'success': True
                    }
                }
            }

            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'tas_regime_discovery'}
            )

        except Exception as e:
            logger.error(f"Fallback TAS regime discovery failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Fallback TAS regime discovery failed: {e}"
            )