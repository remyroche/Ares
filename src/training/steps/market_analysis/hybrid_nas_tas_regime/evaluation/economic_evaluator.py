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
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

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
        tprint_info("🚀 Initializing Economic Regime Evaluator")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize significance types
        tprint_debug("📊 Initializing significance types...")
        self.significance_types = config.get('significance_types', [
            EconomicSignificanceType.VOLATILITY_REGIME.value,
            EconomicSignificanceType.TREND_STRENGTH.value,
            EconomicSignificanceType.VOLUME_PROFILE.value,
            EconomicSignificanceType.CORRELATION_STRUCTURE.value,
            EconomicSignificanceType.MARKET_EFFICIENCY.value,
            EconomicSignificanceType.LIQUIDITY_REGIME.value
        ])
        tprint_success(f"✅ Significance types initialized: {len(self.significance_types)}")

        tprint_debug("⚙️ Setting thresholds...")
        self.min_significance_score = config.get('min_significance_score', 0.7)
        self.volatility_threshold = config.get('volatility_threshold', 0.3)
        self.trend_threshold = config.get('trend_threshold', 0.5)
        self.efficiency_threshold = config.get('efficiency_threshold', 0.6)
        tprint_success("✅ Thresholds configured")

        tprint_success("✅ Economic Regime Evaluator initialized")
        tprint_info(f"   Significance types: {len(self.significance_types)}")
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

            # 7. Momentum regime significance
            if EconomicSignificanceType.MOMENTUM_REGIME.value in self.significance_types:
                momentum_score = self._evaluate_momentum_regime(regime_data)
                significance_scores.append(momentum_score)

            # 8. Volume-momentum regime significance
            if EconomicSignificanceType.VOLUME_MOMENTUM.value in self.significance_types:
                volume_momentum_score = self._evaluate_volume_momentum_regime(regime_data)
                significance_scores.append(volume_momentum_score)

            # 9. Price action regime significance
            if EconomicSignificanceType.PRICE_ACTION.value in self.significance_types:
                price_action_score = self._evaluate_price_action_regime(regime_data)
                significance_scores.append(price_action_score)

            # 10. Market microstructure regime significance
            if EconomicSignificanceType.MARKET_MICROSTRUCTURE.value in self.significance_types:
                microstructure_score = self._evaluate_market_microstructure_regime(regime_data)
                significance_scores.append(microstructure_score)

            # 11. Short-term momentum regime significance (15m focused)
            if EconomicSignificanceType.SHORT_TERM_MOMENTUM.value in self.significance_types:
                short_term_momentum_score = self._evaluate_short_term_momentum_regime(regime_data)
                significance_scores.append(short_term_momentum_score)

            # 12. Intra-bar patterns regime significance (15m bar analysis)
            if EconomicSignificanceType.INTRA_BAR_PATTERNS.value in self.significance_types:
                intra_bar_patterns_score = self._evaluate_intra_bar_patterns_regime(regime_data)
                significance_scores.append(intra_bar_patterns_score)

            # 13. Microstructure patterns regime significance (15m microstructure)
            if EconomicSignificanceType.MICROSTRUCTURE_PATTERNS.value in self.significance_types:
                microstructure_patterns_score = self._evaluate_microstructure_patterns_regime(regime_data)
                significance_scores.append(microstructure_patterns_score)

            # 14. Sector rotation regime significance
            if EconomicSignificanceType.SECTOR_ROTATION.value in self.significance_types:
                sector_rotation_score = self._evaluate_sector_rotation_regime(regime_data)
                significance_scores.append(sector_rotation_score)

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

            # Enhanced scoring with momentum and volume considerations
            base_score = self._calculate_base_volatility_score(relative_volatility)

            # Momentum factor - high volatility with trend is more significant
            momentum_factor = self._calculate_momentum_factor(regime_data)

            # Volume factor - high volatility with high volume is more significant
            volume_factor = self._calculate_volume_factor(regime_data)

            # Combine factors
            enhanced_score = (
                0.5 * base_score +
                0.3 * momentum_factor +
                0.2 * volume_factor
            )

            return min(enhanced_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Volatility evaluation failed: {e}")
            return 0.5

    def _calculate_base_volatility_score(self, relative_volatility: float) -> float:
        """Calculate base volatility score."""
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

    def _calculate_momentum_factor(self, regime_data: pd.DataFrame) -> float:
        """Calculate momentum factor for volatility significance."""
        try:
            prices = regime_data['close'].values

            if len(prices) < 20:
                return 0.5

            # Calculate momentum across different periods
            momentum_scores = []

            for period in [5, 10, 20]:
                if len(prices) > period:
                    # Price change over period
                    momentum = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
                    momentum_strength = abs(np.tanh(momentum))  # Normalize to [0, 1]
                    momentum_scores.append(momentum_strength)

            # High momentum with high volatility is significant
            avg_momentum = np.mean(momentum_scores) if momentum_scores else 0.5
            return min(avg_momentum * 1.5, 1.0)  # Boost momentum factor

        except Exception as e:
            self.logger.warning(f"Momentum factor calculation failed: {e}")
            return 0.5

    def _calculate_volume_factor(self, regime_data: pd.DataFrame) -> float:
        """Calculate volume factor for volatility significance."""
        try:
            if 'volume' not in regime_data.columns:
                return 0.5

            volume = regime_data['volume'].values
            returns = regime_data['close'].pct_change().fillna(0).values

            if len(volume) < 10:
                return 0.5

            # Volume-volatility correlation
            volume_volatility_corr = np.corrcoef(volume[1:], np.abs(returns[1:]))[0, 1]

            # Volume trend
            volume_trend = pd.Series(volume).pct_change().fillna(0).mean()

            # High volume with high volatility correlation is significant
            volume_significance = abs(volume_volatility_corr) * 0.7 + min(abs(volume_trend) * 2, 0.3)

            return min(volume_significance, 1.0)

        except Exception as e:
            self.logger.warning(f"Volume factor calculation failed: {e}")
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

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
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

    def _evaluate_momentum_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate momentum regime significance."""
        try:
            prices = regime_data['close'].values

            if len(prices) < 20:
                return 0.5

            # Calculate momentum across different periods
            momentum_scores = []

            for period in self.config.get('momentum_periods', [5, 10, 20, 50]):
                if len(prices) > period:
                    # Price momentum
                    momentum = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
                    momentum_strength = abs(np.tanh(momentum))  # Normalize to [0, 1]
                    momentum_scores.append(momentum_strength)

                    # Rate of change momentum
                    roc = np.mean(np.diff(np.log(prices[-period:])) * 100)
                    roc_strength = min(abs(roc) / 5.0, 1.0)  # Scale to [0, 1]
                    momentum_scores.append(roc_strength)

            if not momentum_scores:
                return 0.5

            avg_momentum = np.mean(momentum_scores)

            # Momentum regimes are significant when momentum is strong and persistent
            momentum_threshold = self.config.get('momentum_threshold', 0.7)

            if avg_momentum > momentum_threshold:
                score = min(avg_momentum * 1.2, 1.0)  # Strong momentum regime
            elif avg_momentum > 0.4:
                score = avg_momentum * 0.8  # Moderate momentum regime
            else:
                score = avg_momentum * 0.5  # Weak momentum regime

            return score

        except Exception as e:
            self.logger.warning(f"Momentum regime evaluation failed: {e}")
            return 0.5

    def _evaluate_volume_momentum_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate volume-momentum regime significance."""
        try:
            if 'volume' not in regime_data.columns:
                return 0.5

            prices = regime_data['close'].values
            volume = regime_data['volume'].values

            if len(prices) < 20 or len(volume) < 20:
                return 0.5

            # Volume-price momentum correlation
            price_returns = np.diff(prices, prepend=prices[0])
            volume_changes = np.diff(volume, prepend=volume[0])

            # Correlation between volume changes and price movements
            correlation = np.corrcoef(volume_changes[1:], price_returns[1:])[0, 1]

            # Volume momentum (trend in volume)
            volume_momentum = np.mean(volume_changes) / (np.mean(volume) + 1e-8)

            # Price momentum
            price_momentum = (prices[-1] - prices[0]) / (prices[0] + 1e-8)

            # Combined volume-momentum significance
            correlation_factor = abs(correlation) * 0.6  # Correlation significance
            volume_trend_factor = min(abs(volume_momentum) * 2, 0.4)  # Volume trend significance

            # Volume-momentum regimes are significant when there's strong correlation
            volume_threshold = self.config.get('volume_threshold', 0.6)

            if abs(correlation) > volume_threshold:
                score = min(correlation_factor + volume_trend_factor + 0.5, 1.0)
            else:
                score = correlation_factor + volume_trend_factor

            return score

        except Exception as e:
            self.logger.warning(f"Volume-momentum regime evaluation failed: {e}")
            return 0.5

    def _evaluate_price_action_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate price action regime significance."""
        try:
            if len(regime_data) < 10:
                return 0.5

            # Candlestick pattern analysis
            open_prices = regime_data['open'].values
            high_prices = regime_data['high'].values
            low_prices = regime_data['low'].values
            close_prices = regime_data['close'].values

            # Body size analysis
            body_sizes = abs(close_prices - open_prices) / (high_prices - low_prices + 1e-8)
            avg_body_size = np.mean(body_sizes)

            # Shadow analysis
            upper_shadows = (high_prices - np.maximum(open_prices, close_prices)) / (high_prices - low_prices + 1e-8)
            lower_shadows = (np.minimum(open_prices, close_prices) - low_prices) / (high_prices - low_prices + 1e-8)

            avg_upper_shadow = np.mean(upper_shadows)
            avg_lower_shadow = np.mean(lower_shadows)

            # Price action significance factors
            body_factor = min(avg_body_size * 3, 1.0)  # Larger bodies = more decisive
            shadow_factor = min((avg_upper_shadow + avg_lower_shadow) / 2 * 2, 1.0)  # More shadows = more indecision

            # Price action regimes are significant when there's clear directional movement
            price_action_score = 0.6 * body_factor + 0.4 * (1 - shadow_factor)

            return price_action_score

        except Exception as e:
            self.logger.warning(f"Price action regime evaluation failed: {e}")
            return 0.5

    def _evaluate_market_microstructure_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate market microstructure regime significance."""
        try:
            if len(regime_data) < 20:
                return 0.5

            # Bid-ask spread estimation (using high-low as proxy)
            spreads = (regime_data['high'] - regime_data['low']) / regime_data['close']
            avg_spread = np.mean(spreads)

            # Order flow imbalance (using volume as proxy)
            if 'volume' in regime_data.columns:
                volume = regime_data['volume'].values
                price_changes = np.diff(regime_data['close'].values, prepend=regime_data['close'].values[0])

                # Volume-price correlation as order flow proxy
                order_flow_corr = np.corrcoef(volume[1:], np.abs(price_changes[1:]))[0, 1]

                # Market impact estimation
                market_impact = np.mean(np.abs(price_changes) / (volume[1:] + 1e-8))
            else:
                order_flow_corr = 0.5
                market_impact = 0.01

            # Microstructure significance
            spread_factor = min(avg_spread * 20, 1.0)  # Normalize spread
            order_flow_factor = abs(order_flow_corr) * 0.6  # Order flow significance
            impact_factor = min(market_impact * 100, 0.4)  # Market impact significance

            microstructure_score = spread_factor + order_flow_factor + impact_factor

            return min(microstructure_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Market microstructure regime evaluation failed: {e}")
            return 0.5

    def _evaluate_short_term_momentum_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate short-term momentum regime significance for 15m trading."""
        try:
            prices = regime_data['close'].values

            if len(prices) < 10:  # Need at least 10 bars for 15m analysis
                return 0.5

            # Calculate short-term momentum (1-10 periods = 15m to 2.5h)
            momentum_scores = []

            for period in self.config.get('momentum_periods', [1, 2, 5, 10]):
                if len(prices) > period:
                    # Very short-term momentum (15m to 2.5h)
                    momentum = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
                    momentum_strength = abs(np.tanh(momentum))  # Normalize to [0, 1]
                    momentum_scores.append(momentum_strength)

                    # Rate of change for the period
                    if period > 1:
                        roc = (prices[-1] - prices[-period]) / (prices[-period] + 1e-8)
                        roc_strength = abs(np.tanh(roc))
                        momentum_scores.append(roc_strength)

            if not momentum_scores:
                return 0.5

            avg_momentum = np.mean(momentum_scores)

            # Short-term momentum regimes are significant when momentum is strong and recent
            momentum_threshold = self.config.get('momentum_threshold', 0.7)

            if avg_momentum > momentum_threshold:
                score = min(avg_momentum * 1.3, 1.0)  # Strong short-term momentum regime
            elif avg_momentum > 0.5:
                score = avg_momentum * 0.9  # Moderate short-term momentum regime
            else:
                score = avg_momentum * 0.6  # Weak momentum regime

            return score

        except Exception as e:
            self.logger.warning(f"Short-term momentum regime evaluation failed: {e}")
            return 0.5

    def _evaluate_intra_bar_patterns_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate intra-bar patterns regime significance for 15m bars."""
        try:
            if len(regime_data) < 5:
                return 0.5

            # Analyze price action within 15m bars
            open_prices = regime_data['open'].values
            high_prices = regime_data['high'].values
            low_prices = regime_data['low'].values
            close_prices = regime_data['close'].values

            # Body size analysis (relative to total range)
            body_sizes = abs(close_prices - open_prices) / (high_prices - low_prices + 1e-8)
            avg_body_size = np.mean(body_sizes)

            # Shadow analysis
            upper_shadows = (high_prices - np.maximum(open_prices, close_prices)) / (high_prices - low_prices + 1e-8)
            lower_shadows = (np.minimum(open_prices, close_prices) - low_prices) / (high_prices - low_prices + 1e-8)

            avg_upper_shadow = np.mean(upper_shadows)
            avg_lower_shadow = np.mean(lower_shadows)

            # Price position within the bar (close relative to range)
            price_positions = (close_prices - low_prices) / (high_prices - low_prices + 1e-8)
            avg_price_position = np.mean(price_positions)

            # Intra-bar pattern significance factors
            body_factor = min(avg_body_size * 4, 1.0)  # Larger bodies = more decisive
            shadow_factor = min((avg_upper_shadow + avg_lower_shadow) / 2 * 2, 1.0)  # More shadows = more indecision
            position_factor = abs(avg_price_position - 0.5) * 2  # Extreme positions are more significant

            # Intra-bar patterns are significant for short-term trading
            intra_bar_score = 0.5 * body_factor + 0.3 * (1 - shadow_factor) + 0.2 * position_factor

            return min(intra_bar_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Intra-bar patterns regime evaluation failed: {e}")
            return 0.5

    def _evaluate_microstructure_patterns_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate market microstructure patterns for 15m trading."""
        try:
            if len(regime_data) < 10:
                return 0.5

            # Analyze microstructure patterns within 15m bars
            spreads = (regime_data['high'] - regime_data['low']) / regime_data['close']
            avg_spread = np.mean(spreads)
            spread_volatility = np.std(spreads)

            # Volume-based microstructure (if available)
            if 'volume' in regime_data.columns:
                volume = regime_data['volume'].values
                price_changes = np.diff(regime_data['close'].values, prepend=regime_data['close'].values[0])

                # Volume-price impact correlation
                volume_price_impact = np.corrcoef(volume[1:], np.abs(price_changes[1:]))[0, 1]

                # Volume concentration (high volume periods)
                volume_mean = np.mean(volume)
                volume_std = np.std(volume)
                volume_concentration = min(volume_std / volume_mean if volume_mean > 0 else 0, 2.0) / 2.0
            else:
                volume_price_impact = 0.5
                volume_concentration = 0.5

            # Microstructure significance factors
            spread_factor = min(avg_spread * 20, 1.0)  # Normalize spread impact
            impact_factor = abs(volume_price_impact) * 0.7  # Volume-price impact significance
            concentration_factor = volume_concentration  # Volume concentration significance

            # Microstructure patterns are crucial for short-term trading
            microstructure_score = 0.4 * spread_factor + 0.4 * impact_factor + 0.2 * concentration_factor

            return min(microstructure_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Microstructure patterns regime evaluation failed: {e}")
            return 0.5

    def _evaluate_sector_rotation_regime(self, regime_data: pd.DataFrame) -> float:
        """Evaluate sector rotation regime significance for 15m trading."""
        try:
            prices = regime_data['close'].values
            returns = np.diff(prices, prepend=prices[0])

            if len(returns) < 20:  # Need at least 20 bars for 15m analysis
                return 0.5

            # Detect short-term rotation patterns (15m to 5h timeframe)
            # Look for rapid changes in price direction and volatility

            # Short-term volatility changes (5-10 bar windows)
            short_volatility = pd.Series(np.abs(returns)).rolling(window=5, min_periods=3).std().fillna(0.01).values

            # Detect rapid volatility regime changes
            volatility_changes = np.abs(np.diff(short_volatility, prepend=short_volatility[0]))

            # Count significant rotation events
            rotation_threshold = np.percentile(volatility_changes, 70)  # Use 70th percentile for 15m data
            rotation_events = np.sum(volatility_changes > rotation_threshold)

            # Normalize by data length
            rotation_frequency = rotation_events / len(volatility_changes) if len(volatility_changes) > 0 else 0

            # Price direction changes (rapid reversals)
            price_direction = np.sign(returns)
            direction_changes = np.sum(np.abs(np.diff(price_direction, prepend=price_direction[0])) > 0)

            direction_change_rate = direction_changes / len(price_direction) if len(price_direction) > 0 else 0

            # Short-term sector rotation significance
            # More frequent rotations indicate active sector rotation
            rotation_score = min(rotation_frequency * 15, 1.0)  # Scale for 15m trading
            direction_score = min(direction_change_rate * 3, 1.0)  # Scale direction changes

            sector_rotation_score = 0.6 * rotation_score + 0.4 * direction_score

            return min(sector_rotation_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Sector rotation regime evaluation failed: {e}")
            return 0.5


def create_economic_evaluator(config: Dict[str, Any]) -> EconomicRegimeEvaluator:
    """Create economic regime evaluator."""
    return EconomicRegimeEvaluator(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
