"""
Economic Regime Evaluator

Evaluates the economic significance and financial relevance of detected regimes.
This provides the economic and financial relevance that makes the hybrid system
superior to pure HMM-based clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime

from ..config.hybrid_regime_config import HybridRegimeConfig, EconomicSignificanceType

logger = logging.getLogger(__name__)


class EconomicRegimeEvaluator:
    """
    Economic Regime Evaluator

    Evaluates market regimes for economic significance and financial relevance,
    ensuring that detected regimes have meaningful economic interpretation
    and trading value.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize economic regime evaluator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize significance types
        self.significance_types = config.get('significance_types', [
            EconomicSignificanceType.VOLATILITY_REGIME.value,
            EconomicSignificanceType.TREND_STRENGTH.value,
            EconomicSignificanceType.VOLUME_PROFILE.value,
            EconomicSignificanceType.CORRELATION_STRUCTURE.value,
            EconomicSignificanceType.MARKET_EFFICIENCY.value,
            EconomicSignificanceType.LIQUIDITY_REGIME.value
        ])

        self.min_significance_score = config.get('min_significance_score', 0.7)
        self.volatility_threshold = config.get('volatility_threshold', 0.3)
        self.trend_threshold = config.get('trend_threshold', 0.5)
        self.efficiency_threshold = config.get('efficiency_threshold', 0.6)

        self.logger.info("✅ Economic Regime Evaluator initialized")
        self.logger.info(f"   Significance types: {len(self.significance_types)}")
        self.logger.info(f"   Min significance score: {self.min_significance_score}")

    def evaluate_regimes(self,
                        market_data: pd.DataFrame,
                        regime_labels: np.ndarray,
                        regime_probabilities: np.ndarray) -> np.ndarray:
        """
        Evaluate economic significance of each regime.

        Args:
            market_data: Market data
            regime_labels: Regime predictions for each data point
            regime_probabilities: Regime probabilities for each data point

        Returns:
            Array of economic significance scores for each regime
        """
        try:
            n_regimes = len(set(regime_labels))
            significance_scores = np.zeros(n_regimes)

            for regime_id in range(n_regimes):
                regime_mask = regime_labels == regime_id
                if np.sum(regime_mask) > 0:
                    regime_data = market_data[regime_mask]
                    regime_probs = regime_probabilities[regime_mask, regime_id]

                    # Calculate comprehensive economic significance
                    significance = self._calculate_regime_significance(
                        regime_data, regime_probs, regime_id
                    )

                    significance_scores[regime_id] = significance

            self.logger.info(f"   Economic significance scores: {significance_scores}")
            return significance_scores

        except Exception as e:
            self.logger.error(f"Economic evaluation failed: {e}")
            return np.full(len(set(regime_labels)), 0.5)

    def _calculate_regime_significance(self,
                                     regime_data: pd.DataFrame,
                                     regime_probs: np.ndarray,
                                     regime_id: int) -> float:
        """Calculate economic significance for a single regime."""
        try:
            significance_scores = []

            # 1. Volatility regime significance
            if EconomicSignificanceType.VOLATILITY_REGIME.value in self.significance_types:
                vol_score = self._evaluate_volatility_regime(regime_data)
                significance_scores.append(vol_score)

            # 2. Trend strength significance
            if EconomicSignificanceType.TREND_STRENGTH.value in self.significance_types:
                trend_score = self._evaluate_trend_strength(regime_data)
                significance_scores.append(trend_score)

            # 3. Volume profile significance
            if EconomicSignificanceType.VOLUME_PROFILE.value in self.significance_types:
                volume_score = self._evaluate_volume_profile(regime_data)
                significance_scores.append(volume_score)

            # 4. Correlation structure significance
            if EconomicSignificanceType.CORRELATION_STRUCTURE.value in self.significance_types:
                corr_score = self._evaluate_correlation_structure(regime_data)
                significance_scores.append(corr_score)

            # 5. Market efficiency significance
            if EconomicSignificanceType.MARKET_EFFICIENCY.value in self.significance_types:
                efficiency_score = self._evaluate_market_efficiency(regime_data)
                significance_scores.append(efficiency_score)

            # 6. Liquidity regime significance
            if EconomicSignificanceType.LIQUIDITY_REGIME.value in self.significance_types:
                liquidity_score = self._evaluate_liquidity_regime(regime_data)
                significance_scores.append(liquidity_score)

            # Calculate average significance
            if significance_scores:
                avg_significance = np.mean(significance_scores)
            else:
                avg_significance = 0.5

            # Factor in regime probability confidence
            prob_confidence = np.mean(regime_probs)
            final_significance = 0.8 * avg_significance + 0.2 * prob_confidence

            return final_significance

        except Exception as e:
            self.logger.warning(f"Significance calculation failed for regime {regime_id}: {e}")
            return 0.5

    def _evaluate_volatility_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate volatility regime significance."""
        try:
            # Calculate regime volatility
            returns = regime_data['close'].pct_change().dropna()
            regime_volatility = returns.std() if len(returns) > 0 else 0

            # Compare to overall market volatility (using first 1000 points as reference)
            overall_returns = regime_data['close'].pct_change().dropna()
            if len(overall_returns) > 1000:
                reference_returns = overall_returns.iloc[:1000]
                reference_volatility = reference_returns.std()
            else:
                reference_volatility = overall_returns.std()

            # Calculate relative volatility
            if reference_volatility > 0:
                relative_volatility = regime_volatility / reference_volatility
            else:
                relative_volatility = 1.0

            # Score based on deviation from average volatility
            if relative_volatility > self.volatility_threshold:
                # High volatility regime
                score = min(relative_volatility, 2.0) / 2.0  # Cap at 2x average
            elif relative_volatility < (1.0 / self.volatility_threshold):
                # Low volatility regime
                score = (1.0 / relative_volatility) / 2.0  # Cap at 1/(threshold) times average
            else:
                # Normal volatility
                score = 0.5

            return score

        except Exception as e:
            self.logger.warning(f"Volatility evaluation failed: {e}")
            return 0.5

    def _evaluate_trend_strength(self, regime_data: pd.DataFrame) -> float:
        """Evaluate trend strength significance."""
        try:
            # Calculate trend strength using linear regression
            prices = regime_data['close'].values
            if len(prices) < 10:
                return 0.5

            # Simple linear trend
            x = np.arange(len(prices))
            y = prices

            # Calculate R-squared of linear trend
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            r_squared = r_value ** 2

            # Strong trend if R-squared is high and significant
            if p_value < 0.05 and r_squared > 0.1:
                trend_score = min(r_squared * 2.0, 1.0)  # Scale to [0, 1]
            else:
                trend_score = 0.3  # Weak or no trend

            return trend_score

        except Exception as e:
            self.logger.warning(f"Trend strength evaluation failed: {e}")
            return 0.5

    def _evaluate_volume_profile(self, regime_data: pd.DataFrame) -> float:
        """Evaluate volume profile significance."""
        try:
            volume = regime_data.get('volume', np.ones(len(regime_data))).values

            # Calculate volume characteristics
            avg_volume = np.mean(volume)
            volume_std = np.std(volume)
            volume_cv = volume_std / avg_volume if avg_volume > 0 else 0

            # High volume regime if coefficient of variation is low and average is high
            # Low volume regime if coefficient of variation is high and average is low

            # Compare to overall volume profile
            overall_avg_volume = np.mean(volume)  # This is the same as avg_volume for single regime
            overall_volume_std = np.std(volume)

            if overall_volume_std > 0:
                relative_volume = avg_volume / overall_avg_volume
                relative_variation = volume_cv / (overall_volume_std / overall_avg_volume)
            else:
                relative_volume = 1.0
                relative_variation = 1.0

            # Score based on distinctiveness
            if relative_volume > 1.5 and relative_variation < 0.8:
                # High volume, stable regime
                score = 0.9
            elif relative_volume < 0.7 and relative_variation > 1.2:
                # Low volume, variable regime
                score = 0.8
            else:
                score = 0.5

            return score

        except Exception as e:
            self.logger.warning(f"Volume profile evaluation failed: {e}")
            return 0.5

    def _evaluate_correlation_structure(self, regime_data: pd.DataFrame) -> float:
        """Evaluate correlation structure significance."""
        try:
            # Calculate correlations between OHLCV features
            numeric_cols = ['open', 'high', 'low', 'close']
            if 'volume' in regime_data.columns:
                numeric_cols.append('volume')

            if len(numeric_cols) < 2:
                return 0.5

            correlations = regime_data[numeric_cols].corr()

            # Calculate average absolute correlation
            avg_correlation = np.mean(np.abs(correlations.values))
            # Remove diagonal elements (self-correlation)
            n = len(correlations)
            avg_correlation = (avg_correlation * n * n - n) / (n * n - n)

            # High correlation indicates structured regime
            # Low correlation indicates random/noisy regime
            if avg_correlation > 0.7:
                score = 0.9  # Highly structured
            elif avg_correlation > 0.4:
                score = 0.7  # Moderately structured
            else:
                score = 0.3  # Weakly structured

            return score

        except Exception as e:
            self.logger.warning(f"Correlation structure evaluation failed: {e}")
            return 0.5

    def _evaluate_market_efficiency(self, regime_data: pd.DataFrame) -> float:
        """Evaluate market efficiency significance."""
        try:
            # Calculate returns autocorrelation (efficient markets have low autocorrelation)
            returns = regime_data['close'].pct_change().dropna()
            if len(returns) < 10:
                return 0.5

            # Calculate autocorrelation at lag 1
            autocorr = returns.autocorr(lag=1)

            # Efficient market should have low autocorrelation
            # Inefficient market has high autocorrelation (predictable patterns)
            abs_autocorr = abs(autocorr)

            if abs_autocorr < 0.1:
                # Efficient regime
                score = 0.8
            elif abs_autocorr > 0.3:
                # Inefficient regime with predictable patterns
                score = 0.7
            else:
                score = 0.5

            return score

        except Exception as e:
            self.logger.warning(f"Market efficiency evaluation failed: {e}")
            return 0.5

    def _evaluate_liquidity_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate liquidity regime significance."""
        try:
            # Calculate liquidity metrics
            high_low_spread = (regime_data['high'] - regime_data['low']) / regime_data['close']
            avg_spread = np.mean(high_low_spread)

            volume = regime_data.get('volume', np.ones(len(regime_data))).values
            price = regime_data['close'].values

            # Calculate Amihud illiquidity measure
            # Illiquidity = |return| / (price * volume)
            returns = np.diff(price, prepend=price[0])
            illiquidity = np.mean(np.abs(returns) / (price * volume))

            # High liquidity: low spread, low illiquidity
            # Low liquidity: high spread, high illiquidity

            if avg_spread < 0.01 and illiquidity < 0.001:
                # High liquidity regime
                score = 0.9
            elif avg_spread > 0.05 or illiquidity > 0.01:
                # Low liquidity regime
                score = 0.8
            else:
                score = 0.5

            return score

        except Exception as e:
            self.logger.warning(f"Liquidity regime evaluation failed: {e}")
            return 0.5


def create_economic_evaluator(config: Dict[str, Any]) -> EconomicRegimeEvaluator:
    """Create economic regime evaluator."""
    return EconomicRegimeEvaluator(config)