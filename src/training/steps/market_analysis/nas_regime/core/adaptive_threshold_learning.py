"""
Adaptive Threshold Learning System

Data-driven approach to determine optimal thresholds for:
- Economic Significance
- Trading Viability
- Regime Stability

Uses historical market data, regime performance, and market conditions
to dynamically adjust thresholds instead of hardcoded values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ThresholdLearningConfig:
    """Configuration for adaptive threshold learning."""
    # Learning parameters
    lookback_periods: int = 1000  # Historical periods for learning
    min_samples_for_learning: int = 100  # Minimum samples required
    learning_frequency: int = 50  # Update thresholds every N periods
    confidence_level: float = 0.95  # Confidence level for threshold determination

    # Market condition adaptation
    volatility_regime_detection: bool = True
    market_stress_detection: bool = True
    liquidity_regime_detection: bool = True

    # Threshold bounds
    economic_significance_bounds: Tuple[float, float] = (0.3, 0.95)
    trading_viability_bounds: Tuple[float, float] = (0.2, 0.9)
    regime_stability_bounds: Tuple[float, float] = (0.4, 0.95)

    # Market condition weights
    bull_market_weight: float = 1.1
    bear_market_weight: float = 0.9
    high_volatility_weight: float = 0.8
    low_volatility_weight: float = 1.2
    high_liquidity_weight: float = 1.1
    low_liquidity_weight: float = 0.9

@dataclass
class AdaptiveThresholds:
    """Learned adaptive thresholds."""
    economic_significance_threshold: float
    trading_viability_threshold: float
    regime_stability_threshold: float

    # Confidence intervals
    economic_confidence_interval: Tuple[float, float]
    trading_confidence_interval: Tuple[float, float]
    stability_confidence_interval: Tuple[float, float]

    # Market condition adjustments
    market_condition_adjustments: Dict[str, float]

    # Learning metadata
    learning_samples: int
    last_updated: pd.Timestamp
    learning_confidence: float

class AdaptiveThresholdLearner:
    """
    Adaptive threshold learning system that determines optimal thresholds
    based on historical market data and regime performance.
    """

    def __init__(self, config: ThresholdLearningConfig = None):
        """Initialize adaptive threshold learner.

        Args:
            config: Threshold learning configuration
        """
        self.config = config or ThresholdLearningConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Learning state
        self.learning_history = []
        self.current_thresholds = None
        self.market_conditions = {}
        self.performance_metrics = {}

        # Models for market condition detection
        self.volatility_model = None
        self.liquidity_model = None
        self.stress_model = None

        self.logger.info("✅ Adaptive Threshold Learner initialized")

    def learn_thresholds(self,
                        market_data: np.ndarray,
                        regime_predictions: np.ndarray,
                        timestamps: Optional[np.ndarray] = None,
                        performance_metrics: Optional[Dict[str, Any]] = None) -> AdaptiveThresholds:
        """
        Learn optimal thresholds from historical data.

        Args:
            market_data: Historical market data (OHLCV)
            regime_predictions: Historical regime predictions
            timestamps: Optional timestamps
            performance_metrics: Optional performance metrics

        Returns:
            Learned adaptive thresholds
        """
        try:
            self.logger.info("🧠 Learning adaptive thresholds from historical data...")

            # Validate input data
            if len(market_data) < self.config.min_samples_for_learning:
                self.logger.warning(f"Insufficient data for learning: {len(market_data)} < {self.config.min_samples_for_learning}")
                return self._get_default_thresholds()

            # Detect market conditions
            market_conditions = self._detect_market_conditions(market_data, timestamps)

            # Calculate regime performance metrics
            regime_metrics = self._calculate_regime_performance_metrics(
                market_data, regime_predictions, timestamps
            )

            # Learn economic significance threshold
            economic_threshold = self._learn_economic_significance_threshold(
                market_data, regime_predictions, regime_metrics, market_conditions
            )

            # Learn trading viability threshold
            trading_threshold = self._learn_trading_viability_threshold(
                market_data, regime_predictions, regime_metrics, market_conditions
            )

            # Learn regime stability threshold
            stability_threshold = self._learn_regime_stability_threshold(
                market_data, regime_predictions, regime_metrics, market_conditions
            )

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                economic_threshold, trading_threshold, stability_threshold, regime_metrics
            )

            # Create adaptive thresholds
            adaptive_thresholds = AdaptiveThresholds(
                economic_significance_threshold=economic_threshold,
                trading_viability_threshold=trading_threshold,
                regime_stability_threshold=stability_threshold,
                economic_confidence_interval=confidence_intervals['economic'],
                trading_confidence_interval=confidence_intervals['trading'],
                stability_confidence_interval=confidence_intervals['stability'],
                market_condition_adjustments=market_conditions,
                learning_samples=len(market_data),
                last_updated=pd.Timestamp.now(),
                learning_confidence=self._calculate_learning_confidence(regime_metrics)
            )

            # Update learning state
            self.current_thresholds = adaptive_thresholds
            self.market_conditions = market_conditions
            self.performance_metrics = regime_metrics

            self.logger.info(f"✅ Adaptive thresholds learned successfully")
            self.logger.info(f"   Economic significance: {economic_threshold:.3f}")
            self.logger.info(f"   Trading viability: {trading_threshold:.3f}")
            self.logger.info(f"   Regime stability: {stability_threshold:.3f}")
            self.logger.info(f"   Learning confidence: {adaptive_thresholds.learning_confidence:.3f}")

            return adaptive_thresholds

        except Exception as e:
            self.logger.error(f"❌ Threshold learning failed: {e}")
            return self._get_default_thresholds()

    def _detect_market_conditions(self, market_data: np.ndarray,
                                timestamps: Optional[np.ndarray]) -> Dict[str, float]:
        """Detect current market conditions."""
        try:
            conditions = {}

            # Calculate basic market metrics
            if market_data.shape[1] >= 4:  # OHLC data
                prices = market_data[:, 3]  # Close prices
                volumes = market_data[:, 4] if market_data.shape[1] > 4 else np.ones(len(prices))
            else:
                prices = market_data[:, 0]  # Use first column as prices
                volumes = np.ones(len(prices))

            # Volatility regime detection
            if self.config.volatility_regime_detection:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                conditions['volatility_regime'] = self._classify_volatility_regime(volatility)
                conditions['volatility_level'] = volatility

            # Market stress detection
            if self.config.market_stress_detection:
                stress_level = self._calculate_market_stress(prices, volumes)
                conditions['stress_level'] = stress_level
                conditions['stress_regime'] = self._classify_stress_regime(stress_level)

            # Liquidity regime detection
            if self.config.liquidity_regime_detection:
                liquidity_level = self._calculate_liquidity_level(volumes, prices)
                conditions['liquidity_level'] = liquidity_level
                conditions['liquidity_regime'] = self._classify_liquidity_regime(liquidity_level)

            # Trend detection
            trend_strength = self._calculate_trend_strength(prices)
            conditions['trend_strength'] = trend_strength
            conditions['trend_direction'] = 1 if trend_strength > 0 else -1

            return conditions

        except Exception as e:
            self.logger.warning(f"Market condition detection failed: {e}")
            return {'volatility_regime': 'normal', 'stress_level': 0.5, 'liquidity_regime': 'normal'}

    def _calculate_regime_performance_metrics(self, market_data: np.ndarray,
                                            regime_predictions: np.ndarray,
                                            timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate regime performance metrics."""
        try:
            metrics = {}

            # Regime distribution
            unique_regimes, regime_counts = np.unique(regime_predictions, return_counts=True)
            metrics['regime_distribution'] = dict(zip(unique_regimes, regime_counts))
            metrics['n_regimes'] = len(unique_regimes)
            metrics['regime_balance'] = np.std(regime_counts) / np.mean(regime_counts)

            # Regime persistence
            regime_changes = np.sum(np.diff(regime_predictions) != 0)
            metrics['regime_changes'] = regime_changes
            metrics['regime_persistence'] = 1 - (regime_changes / len(regime_predictions))

            # Economic significance metrics
            if market_data.shape[1] >= 4:
                prices = market_data[:, 3]  # Close prices
                returns = np.diff(prices) / prices[:-1]

                # Calculate regime-specific returns
                regime_returns = {}
                for regime in unique_regimes:
                    regime_mask = regime_predictions[1:] == regime
                    if np.sum(regime_mask) > 0:
                        regime_returns[regime] = np.mean(returns[regime_mask])

                metrics['regime_returns'] = regime_returns
                metrics['return_dispersion'] = np.std(list(regime_returns.values()))
                metrics['return_consistency'] = 1 - metrics['return_dispersion']

            # Trading viability metrics
            if timestamps is not None:
                regime_durations = self._calculate_regime_durations(regime_predictions, timestamps)
                metrics['avg_regime_duration'] = np.mean(regime_durations)
                metrics['min_regime_duration'] = np.min(regime_durations)
                metrics['max_regime_duration'] = np.max(regime_durations)
                metrics['duration_consistency'] = 1 - (np.std(regime_durations) / np.mean(regime_durations))

            return metrics

        except Exception as e:
            self.logger.warning(f"Performance metrics calculation failed: {e}")
            return {'n_regimes': 1, 'regime_persistence': 0.5, 'return_consistency': 0.5}

    def _learn_economic_significance_threshold(self, market_data: np.ndarray,
                                             regime_predictions: np.ndarray,
                                             regime_metrics: Dict[str, Any],
                                             market_conditions: Dict[str, float]) -> float:
        """Learn optimal economic significance threshold."""
        try:
            # Base threshold from regime performance
            base_threshold = 0.5

            # Adjust based on return consistency
            if 'return_consistency' in regime_metrics:
                base_threshold += regime_metrics['return_consistency'] * 0.3

            # Adjust based on regime balance
            if 'regime_balance' in regime_metrics:
                balance_factor = 1 - regime_metrics['regime_balance']
                base_threshold += balance_factor * 0.2

            # Adjust based on market conditions
            if 'volatility_regime' in market_conditions:
                if market_conditions['volatility_regime'] == 'high':
                    base_threshold *= self.config.high_volatility_weight
                elif market_conditions['volatility_regime'] == 'low':
                    base_threshold *= self.config.low_volatility_weight

            if 'stress_level' in market_conditions:
                stress_factor = 1 - market_conditions['stress_level']
                base_threshold *= stress_factor

            # Apply bounds
            base_threshold = np.clip(
                base_threshold,
                self.config.economic_significance_bounds[0],
                self.config.economic_significance_bounds[1]
            )

            return base_threshold

        except Exception as e:
            self.logger.warning(f"Economic significance threshold learning failed: {e}")
            return 0.8

    def _learn_trading_viability_threshold(self, market_data: np.ndarray,
                                         regime_predictions: np.ndarray,
                                         regime_metrics: Dict[str, Any],
                                         market_conditions: Dict[str, float]) -> float:
        """Learn optimal trading viability threshold."""
        try:
            # Base threshold from regime duration and consistency
            base_threshold = 0.5

            # Adjust based on regime duration
            if 'avg_regime_duration' in regime_metrics:
                duration_factor = min(regime_metrics['avg_regime_duration'] / 100, 1.0)
                base_threshold += duration_factor * 0.3

            # Adjust based on duration consistency
            if 'duration_consistency' in regime_metrics:
                base_threshold += regime_metrics['duration_consistency'] * 0.2

            # Adjust based on liquidity conditions
            if 'liquidity_regime' in market_conditions:
                if market_conditions['liquidity_regime'] == 'high':
                    base_threshold *= self.config.high_liquidity_weight
                elif market_conditions['liquidity_regime'] == 'low':
                    base_threshold *= self.config.low_liquidity_weight

            # Adjust based on trend strength
            if 'trend_strength' in market_conditions:
                trend_factor = abs(market_conditions['trend_strength'])
                base_threshold += trend_factor * 0.1

            # Apply bounds
            base_threshold = np.clip(
                base_threshold,
                self.config.trading_viability_bounds[0],
                self.config.trading_viability_bounds[1]
            )

            return base_threshold

        except Exception as e:
            self.logger.warning(f"Trading viability threshold learning failed: {e}")
            return 0.7

    def _learn_regime_stability_threshold(self, market_data: np.ndarray,
                                        regime_predictions: np.ndarray,
                                        regime_metrics: Dict[str, Any],
                                        market_conditions: Dict[str, float]) -> float:
        """Learn optimal regime stability threshold."""
        try:
            # Base threshold from regime persistence
            base_threshold = 0.5

            # Adjust based on regime persistence
            if 'regime_persistence' in regime_metrics:
                base_threshold += regime_metrics['regime_persistence'] * 0.4

            # Adjust based on regime balance
            if 'regime_balance' in regime_metrics:
                balance_factor = 1 - regime_metrics['regime_balance']
                base_threshold += balance_factor * 0.2

            # Adjust based on market stress
            if 'stress_level' in market_conditions:
                stress_factor = 1 - market_conditions['stress_level']
                base_threshold *= stress_factor

            # Adjust based on volatility
            if 'volatility_level' in market_conditions:
                volatility_factor = 1 - min(market_conditions['volatility_level'] * 10, 1.0)
                base_threshold *= volatility_factor

            # Apply bounds
            base_threshold = np.clip(
                base_threshold,
                self.config.regime_stability_bounds[0],
                self.config.regime_stability_bounds[1]
            )

            return base_threshold

        except Exception as e:
            self.logger.warning(f"Regime stability threshold learning failed: {e}")
            return 0.8

    def _calculate_confidence_intervals(self, economic_threshold: float,
                                      trading_threshold: float,
                                      stability_threshold: float,
                                      regime_metrics: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for thresholds."""
        try:
            # Calculate confidence based on regime metrics
            confidence_factor = 0.1

            if 'regime_persistence' in regime_metrics:
                confidence_factor += regime_metrics['regime_persistence'] * 0.1

            if 'return_consistency' in regime_metrics:
                confidence_factor += regime_metrics['return_consistency'] * 0.1

            # Calculate intervals
            economic_ci = (
                max(0, economic_threshold - confidence_factor),
                min(1, economic_threshold + confidence_factor)
            )

            trading_ci = (
                max(0, trading_threshold - confidence_factor),
                min(1, trading_threshold + confidence_factor)
            )

            stability_ci = (
                max(0, stability_threshold - confidence_factor),
                min(1, stability_threshold + confidence_factor)
            )

            return {
                'economic': economic_ci,
                'trading': trading_ci,
                'stability': stability_ci
            }

        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return {
                'economic': (0.7, 0.9),
                'trading': (0.6, 0.8),
                'stability': (0.7, 0.9)
            }

    def _calculate_learning_confidence(self, regime_metrics: Dict[str, Any]) -> float:
        """Calculate confidence in the learned thresholds."""
        try:
            confidence = 0.5

            # Increase confidence based on regime quality
            if 'regime_persistence' in regime_metrics:
                confidence += regime_metrics['regime_persistence'] * 0.2

            if 'return_consistency' in regime_metrics:
                confidence += regime_metrics['return_consistency'] * 0.2

            if 'duration_consistency' in regime_metrics:
                confidence += regime_metrics['duration_consistency'] * 0.1

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.warning(f"Learning confidence calculation failed: {e}")
            return 0.5

    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime."""
        if volatility > 0.02:
            return 'high'
        elif volatility < 0.005:
            return 'low'
        else:
            return 'normal'

    def _classify_stress_regime(self, stress_level: float) -> str:
        """Classify market stress regime."""
        if stress_level > 0.7:
            return 'high'
        elif stress_level < 0.3:
            return 'low'
        else:
            return 'normal'

    def _classify_liquidity_regime(self, liquidity_level: float) -> str:
        """Classify liquidity regime."""
        if liquidity_level > 0.7:
            return 'high'
        elif liquidity_level < 0.3:
            return 'low'
        else:
            return 'normal'

    def _calculate_market_stress(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate market stress level."""
        try:
            # Price volatility
            returns = np.diff(prices) / prices[:-1]
            price_stress = np.std(returns)

            # Volume stress
            volume_changes = np.diff(volumes) / (volumes[:-1] + 1e-8)
            volume_stress = np.std(volume_changes)

            # Combined stress
            stress = (price_stress + volume_stress) / 2
            return min(stress, 1.0)

        except Exception as e:
            self.logger.warning(f"Market stress calculation failed: {e}")
            return 0.5

    def _calculate_liquidity_level(self, volumes: np.ndarray, prices: np.ndarray) -> float:
        """Calculate liquidity level."""
        try:
            # Volume consistency
            volume_consistency = 1 - (np.std(volumes) / (np.mean(volumes) + 1e-8))

            # Price-volume relationship
            if len(volumes) > 1 and len(prices) > 1:
                volume_price_corr = np.corrcoef(volumes, prices)[0, 1]
                if np.isnan(volume_price_corr):
                    volume_price_corr = 0
            else:
                volume_price_corr = 0

            # Combined liquidity
            liquidity = (volume_consistency + abs(volume_price_corr)) / 2
            return min(liquidity, 1.0)

        except Exception as e:
            self.logger.warning(f"Liquidity calculation failed: {e}")
            return 0.5

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength."""
        try:
            if len(prices) < 2:
                return 0.0

            # Simple trend calculation
            price_change = (prices[-1] - prices[0]) / prices[0]
            return np.tanh(price_change * 10)  # Normalize to [-1, 1]

        except Exception as e:
            self.logger.warning(f"Trend strength calculation failed: {e}")
            return 0.0

    def _calculate_regime_durations(self, regime_predictions: np.ndarray,
                                  timestamps: np.ndarray) -> np.ndarray:
        """Calculate regime durations."""
        try:
            durations = []
            current_regime = regime_predictions[0]
            start_time = timestamps[0]

            for i in range(1, len(regime_predictions)):
                if regime_predictions[i] != current_regime:
                    duration = timestamps[i] - start_time
                    durations.append(duration.total_seconds() / 60)  # Convert to minutes
                    current_regime = regime_predictions[i]
                    start_time = timestamps[i]

            # Add final duration
            if len(durations) > 0:
                final_duration = timestamps[-1] - start_time
                durations.append(final_duration.total_seconds() / 60)

            return np.array(durations) if durations else np.array([0])

        except Exception as e:
            self.logger.warning(f"Regime duration calculation failed: {e}")
            return np.array([60])  # Default 1 hour

    def _get_default_thresholds(self) -> AdaptiveThresholds:
        """Get default thresholds when learning fails."""
        return AdaptiveThresholds(
            economic_significance_threshold=0.8,
            trading_viability_threshold=0.7,
            regime_stability_threshold=0.8,
            economic_confidence_interval=(0.7, 0.9),
            trading_confidence_interval=(0.6, 0.8),
            stability_confidence_interval=(0.7, 0.9),
            market_condition_adjustments={},
            learning_samples=0,
            last_updated=pd.Timestamp.now(),
            learning_confidence=0.5
        )

    def update_thresholds(self, new_market_data: np.ndarray,
                        new_regime_predictions: np.ndarray,
                        new_timestamps: Optional[np.ndarray] = None) -> AdaptiveThresholds:
        """Update thresholds with new data."""
        try:
            self.logger.info("🔄 Updating adaptive thresholds...")

            # Combine with historical data
            if len(self.learning_history) > 0:
                # Use recent history for updating
                recent_data = self.learning_history[-self.config.lookback_periods:]
                # Implementation would combine recent_data with new data
                pass

            # Learn new thresholds
            new_thresholds = self.learn_thresholds(
                new_market_data, new_regime_predictions, new_timestamps
            )

            # Update learning history
            self.learning_history.append({
                'data': new_market_data,
                'predictions': new_regime_predictions,
                'timestamps': new_timestamps,
                'thresholds': new_thresholds
            })

            # Keep only recent history
            if len(self.learning_history) > self.config.lookback_periods:
                self.learning_history = self.learning_history[-self.config.lookback_periods:]

            return new_thresholds

        except Exception as e:
            self.logger.error(f"❌ Threshold update failed: {e}")
            return self.current_thresholds or self._get_default_thresholds()

    def get_current_thresholds(self) -> Optional[AdaptiveThresholds]:
        """Get current adaptive thresholds."""
        return self.current_thresholds

    def get_threshold_explanations(self) -> Dict[str, str]:
        """Get explanations for current thresholds."""
        if not self.current_thresholds:
            return {"error": "No thresholds learned yet"}

        explanations = {}

        # Economic significance explanation
        economic = self.current_thresholds.economic_significance_threshold
        explanations['economic_significance'] = (
            f"Economic significance threshold: {economic:.3f}. "
            f"This threshold was learned from historical market data and regime performance. "
            f"Higher values indicate more stringent economic relevance requirements."
        )

        # Trading viability explanation
        trading = self.current_thresholds.trading_viability_threshold
        explanations['trading_viability'] = (
            f"Trading viability threshold: {trading:.3f}. "
            f"This threshold was determined based on regime duration, liquidity conditions, "
            f"and market structure. It ensures regimes are actionable for trading decisions."
        )

        # Regime stability explanation
        stability = self.current_thresholds.regime_stability_threshold
        explanations['regime_stability'] = (
            f"Regime stability threshold: {stability:.3f}. "
            f"This threshold was learned from regime persistence patterns and market conditions. "
            f"It ensures detected regimes are stable and persistent."
        )

        return explanations
