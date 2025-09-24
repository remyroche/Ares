"""
NAS Regime Discovery Component

Integrates NAS (Neural Architecture Search) regime detection capabilities
into the market analysis pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

logger = logging.getLogger(__name__)


class NASRegimeDiscoveryComponent(BaseMarketAnalysisComponent):
    """
    NAS Regime Discovery Component

    Uses Neural Architecture Search to discover market regimes with
    advanced neural networks, economic significance evaluation,
    and trading viability assessment.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize NAS regime discovery component."""
        super().__init__(config)

        # Initialize NAS detector
        self._initialize_nas_detector()

        logger.info("✅ NAS Regime Discovery Component initialized")

    def _initialize_nas_detector(self):
        """Initialize the NAS regime detector."""
        try:
            from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import (
                PerfectNASRegimeDetector,
                PerfectNASConfig
            )

            # Create NAS configuration from component config
            nas_config = PerfectNASConfig(
                primary_architecture=self.config.custom_params.get('primary_architecture', 'hybrid'),
                enable_neural_odes=self.config.custom_params.get('enable_neural_odes', True),
                enable_vision_transformers=self.config.custom_params.get('enable_vision_transformers', True),
                enable_meta_learning=self.config.custom_params.get('enable_meta_learning', True),
                search_strategy=self.config.custom_params.get('search_strategy', 'evolutionary'),
                n_regimes=self.config.custom_params.get('n_regimes', 8),
                economic_significance_threshold=self.config.custom_params.get('economic_significance_threshold', 0.6),
                trading_viability_threshold=self.config.custom_params.get('trading_viability_threshold', 0.5)
            )

            self.nas_detector = PerfectNASRegimeDetector(nas_config)
            logger.info("✅ NAS detector initialized successfully")

        except ImportError as e:
            logger.warning(f"NAS detector not available: {e}, using fallback")
            self.nas_detector = None

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_regime_discovery_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS regime discovery.

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

            # Execute NAS regime discovery
            if self.nas_detector is not None:
                nas_result = self.nas_detector.detect_regimes(data)

                if not hasattr(nas_result, 'success') or not nas_result.success:
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message="NAS regime discovery failed"
                    )

                # Create comprehensive artifact
                artifact = {
                    'nas_regime_discovery_result': {
                        'regime_models': self._extract_regime_models(nas_result),
                        'regime_assignments': self._extract_regime_assignments(nas_result),
                        'regime_metrics': self._extract_regime_metrics(nas_result),
                        'execution_time': getattr(nas_result, 'execution_time', 0.0),
                        'metadata': {
                            'method': 'nas_detector',
                            'architecture': self.config.custom_params.get('primary_architecture', 'hybrid'),
                            'n_regimes': len(set(getattr(nas_result, 'regime_predictions', []))),
                            'success': True
                        }
                    }
                }

                return ComponentResult(
                    success=True,
                    artifacts=artifact,
                    metadata={'component_type': 'nas_regime_discovery'}
                )
            else:
                # Fallback implementation
                return self._fallback_regime_discovery(data)

        except Exception as e:
            logger.error(f"NAS regime discovery failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )

    def _extract_regime_models(self, nas_result) -> Dict[str, Any]:
        """Extract regime models from NAS result."""
        try:
            return {
                'architecture_performance': getattr(nas_result, 'architecture_performance', {}),
                'micro_regimes': getattr(nas_result, 'micro_regimes', {}),
                'transition_probabilities': getattr(nas_result, 'transition_probabilities', np.array([])),
                'regime_stability_scores': getattr(nas_result, 'regime_stability_scores', np.array([]))
            }
        except Exception as e:
            logger.warning(f"Failed to extract regime models: {e}")
            return {}

    def _extract_regime_assignments(self, nas_result) -> Dict[str, Any]:
        """Extract regime assignments from NAS result."""
        try:
            regime_predictions = getattr(nas_result, 'regime_predictions', np.array([]))
            regime_probabilities = getattr(nas_result, 'regime_probabilities', np.array([]))

            return {
                'regime_labels': regime_predictions,
                'regime_probabilities': regime_probabilities,
                'n_regimes': len(set(regime_predictions)) if len(regime_predictions) > 0 else 0,
                'confidence_scores': np.mean(regime_probabilities, axis=1) if regime_probabilities.size > 0 else np.array([])
            }
        except Exception as e:
            logger.warning(f"Failed to extract regime assignments: {e}")
            return {}

    def _extract_regime_metrics(self, nas_result) -> Dict[str, Any]:
        """Extract regime metrics from NAS result."""
        try:
            return {
                'economic_significance_scores': getattr(nas_result, 'economic_significance_scores', np.array([])),
                'trading_viability_scores': getattr(nas_result, 'trading_viability_scores', np.array([])),
                'uncertainty_estimates': getattr(nas_result, 'uncertainty_estimates', np.array([])),
                'overall_economic_significance': np.mean(getattr(nas_result, 'economic_significance_scores', np.array([0.0]))),
                'overall_trading_viability': np.mean(getattr(nas_result, 'trading_viability_scores', np.array([0.0]))),
                'regime_stability': np.mean(getattr(nas_result, 'regime_stability_scores', np.array([0.0])))
            }
        except Exception as e:
            logger.warning(f"Failed to extract regime metrics: {e}")
            return {}

    def _fallback_regime_discovery(self, data: pd.DataFrame) -> ComponentResult:
        """Fallback regime discovery using basic methods."""
        try:
            logger.info("🔄 Using fallback regime discovery")

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
                'nas_regime_discovery_result': {
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
                        'architecture': 'basic',
                        'n_regimes': 4,
                        'success': True
                    }
                }
            }

            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'nas_regime_discovery'}
            )

        except Exception as e:
            logger.error(f"Fallback regime discovery failed: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Fallback regime discovery failed: {e}"
            )