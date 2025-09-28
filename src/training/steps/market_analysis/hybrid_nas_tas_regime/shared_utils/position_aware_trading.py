"""
Position-Aware Trading Analysis for Hybrid NAS-TAS Regime Detection.

Provides position-aware trading analysis utilities that work for both long and short positions,
ensuring accurate win rate calculations and trading viability assessments for both TAS and NAS systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


@dataclass
class PositionAwareConfig:
    """Configuration for position-aware trading analysis."""
    minimum_profit_threshold: float = 0.001  # 0.1% minimum profit
    transaction_cost: float = 0.001  # 0.1% transaction cost
    position_holding_periods: List[int] = None  # [1, 5, 10, 20] periods to analyze
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    win_rate_thresholds: Dict[str, float] = None  # Thresholds for different metrics

    def __post_init__(self):
        """Initialize default values."""
        if self.position_holding_periods is None:
            self.position_holding_periods = [1, 5, 10, 20]

        if self.win_rate_thresholds is None:
            self.win_rate_thresholds = {
                'excellent': 0.7,
                'good': 0.6,
                'acceptable': 0.5,
                'poor': 0.4
            }


@dataclass
class PositionAwareResult:
    """Result from position-aware trading analysis."""
    # Position-aware win rates
    overall_win_rate: float
    long_win_rate: float
    short_win_rate: float

    # Position counts
    total_trades: int
    long_trades: int
    short_trades: int

    # Profitability metrics
    long_total_return: float
    short_total_return: float
    long_sharpe_ratio: float
    short_sharpe_ratio: float
    long_max_drawdown: float
    short_max_drawdown: float

    # Position-aware economic significance
    long_economic_significance: float
    short_economic_significance: float

    # Trading viability scores
    long_trading_viability: float
    short_trading_viability: float

    # Position balance metrics
    position_balance_score: float
    position_diversification_benefit: float

    # Analysis metadata
    analysis_timestamp: str
    position_directions_used: bool
    profitable_period_definition: str


class PositionAwareTradingAnalyzer:
    """
    Position-aware trading analyzer that works for both TAS and NAS systems.

    Provides accurate win rate calculations for long and short positions,
    ensuring proper evaluation of trading strategies regardless of position direction.
    """

    def __init__(self, config: PositionAwareConfig = None):
        """Initialize position-aware trading analyzer.

        Args:
            config: Position-aware configuration
        """
        tprint_info("🚀 Initializing Position-Aware Trading Analyzer")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config or PositionAwareConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        tprint_success("✅ Position-Aware Trading Analyzer initialized")
        tprint_info(f"   Minimum profit threshold: {self.config.minimum_profit_threshold}")
        tprint_info(f"   Transaction cost: {self.config.transaction_cost}")
        tprint_info(f"   Position holding periods: {self.config.position_holding_periods}")
        self.logger.info("✅ Position-Aware Trading Analyzer initialized")
        self.logger.info(f"   Minimum profit threshold: {self.config.minimum_profit_threshold}")
        self.logger.info(f"   Transaction cost: {self.config.transaction_cost}")
        self.logger.info(f"   Position holding periods: {self.config.position_holding_periods}")

    def calculate_position_aware_win_rate(
        self,
        returns: np.ndarray,
        position_directions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate position-aware win rates for long and short positions.

        Args:
            returns: Array of percentage returns
            position_directions: Array of position directions
                                (1 = long, -1 = short, 0 = neutral)

        Returns:
            Dict with win rates for longs, shorts, and overall
        """
        try:
            tprint_debug("📊 Calculating position-aware win rates...")
            tprint_debug(f"Returns shape: {returns.shape}")
            tprint_debug(f"Position directions shape: {position_directions.shape}")
            
            results = {
                'overall_win_rate': 0.0,
                'long_win_rate': 0.0,
                'short_win_rate': 0.0,
                'total_trades': len(returns),
                'long_trades': 0,
                'short_trades': 0
            }

            if len(returns) == 0:
                tprint_warning("⚠️ No returns data provided")
                return results

            # Overall win rate (any directional movement after costs)
            adjusted_returns = returns - self.config.transaction_cost
            results['overall_win_rate'] = np.mean(np.abs(adjusted_returns) > self.config.minimum_profit_threshold)

            # Position-specific win rates
            long_mask = position_directions == 1
            short_mask = position_directions == -1

            if np.any(long_mask):
                long_returns = adjusted_returns[long_mask]
                # For longs: positive returns = profit
                results['long_win_rate'] = np.mean(long_returns > self.config.minimum_profit_threshold)
                results['long_trades'] = len(long_returns)

            if np.any(short_mask):
                short_returns = adjusted_returns[short_mask]
                # For shorts: negative returns = profit
                results['short_win_rate'] = np.mean(short_returns < -self.config.minimum_profit_threshold)
                results['short_trades'] = len(short_returns)

            self.logger.info(f"📊 Position-aware win rates calculated:")
            self.logger.info(f"   Overall: {results['overall_win_rate']:.3f}")
            self.logger.info(f"   Long: {results['long_win_rate']:.3f} ({results['long_trades']} trades)")
            self.logger.info(f"   Short: {results['short_win_rate']:.3f} ({results['short_trades']} trades)")

            return results

        except Exception as e:
            self.logger.error(f"Position-aware win rate calculation failed: {e}")
            return {
                'overall_win_rate': 0.5,
                'long_win_rate': 0.5,
                'short_win_rate': 0.5,
                'total_trades': len(returns),
                'long_trades': 0,
                'short_trades': 0
            }

    def analyze_regime_position_performance(
        self,
        market_data: pd.DataFrame,
        regime_labels: np.ndarray,
        position_directions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze position-aware performance per regime.

        Args:
            market_data: Market data DataFrame
            regime_labels: Regime prediction labels
            position_directions: Optional position directions array

        Returns:
            Dict with position-aware analysis per regime
        """
        try:
            results = {
                'overall_analysis': {},
                'regime_analyses': {},
                'position_balance_analysis': {}
            }

            # Overall analysis across all regimes
            returns = market_data['close'].pct_change().dropna().values

            if position_directions is None:
                # Infer positions from returns (positive = long, negative = short)
                position_directions = np.where(returns > 0, 1, -1)
            else:
                position_directions = position_directions[1:]  # Align with returns

            overall_win_rates = self.calculate_position_aware_win_rate(returns, position_directions)
            results['overall_analysis'] = overall_win_rates

            # Per-regime analysis
            unique_regimes = np.unique(regime_labels)

            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                regime_data = market_data[regime_mask]

                if len(regime_data) < 10:
                    continue

                # Get returns and positions for this regime
                regime_returns = regime_data['close'].pct_change().dropna().values

            # Get position directions for this regime - need to align dimensions properly
            # regime_mask has length N, but position_directions has length N-1 after alignment
            # So we need to create a mask of length N-1 for the aligned position_directions
            aligned_regime_mask = regime_mask[1:]  # Align mask with position_directions
            if len(position_directions) != len(aligned_regime_mask):
                # Handle dimension mismatch by ensuring they have the same length
                min_length = min(len(position_directions), len(aligned_regime_mask))
                aligned_regime_mask = aligned_regime_mask[:min_length]
                position_directions = position_directions[:min_length]
            regime_positions = position_directions[aligned_regime_mask]

            # Get aligned regime data for economic significance calculation
            aligned_regime_data = market_data[aligned_regime_mask]

            # Calculate position-aware metrics
            regime_win_rates = self.calculate_position_aware_win_rate(
                regime_returns, regime_positions
            )

            # Add additional regime-specific analysis
            regime_analysis = {
                **regime_win_rates,
                'regime_id': regime_id,
                'regime_duration': len(regime_returns),  # Use returns length for consistency
                'regime_volatility': np.std(regime_returns),
                'position_bias': np.mean(regime_positions),  # >0 = long bias, <0 = short bias
                'economic_significance': self._calculate_regime_economic_significance(
                    aligned_regime_data, regime_positions
                )
            }

            results['regime_analyses'][f"regime_{regime_id}"] = regime_analysis

            # Position balance analysis
            results['position_balance_analysis'] = self._analyze_position_balance(
                position_directions, returns
            )

            return results

        except Exception as e:
            self.logger.error(f"Regime position performance analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_regime_economic_significance(
        self,
        regime_data: pd.DataFrame,
        position_directions: np.ndarray
    ) -> float:
        """
        Calculate position-aware economic significance for a regime.

        Args:
            regime_data: Market data for the regime
            position_directions: Position directions for the regime

        Returns:
            Position-aware economic significance score
        """
        try:
            significance_factors = []

            # Volatility significance (position-neutral)
            returns = regime_data['close'].pct_change().dropna().values
            volatility = np.std(returns) if len(returns) > 0 else 0.01
            vol_significance = min(volatility / 0.05, 1.0)
            significance_factors.append(vol_significance)

            # Position-aware trend significance
            prices = regime_data['close'].values

            # For longs: upward trends are significant
            long_mask = position_directions == 1
            if np.any(long_mask):
                long_prices = prices[long_mask]
                if len(long_prices) > 10:
                    # Calculate upward trend strength for longs
                    long_trend = np.polyfit(range(len(long_prices)), long_prices, 1)[0]
                    long_trend_significance = min(abs(long_trend) / np.mean(long_prices) * 100, 1.0)
                    significance_factors.append(long_trend_significance)

            # For shorts: downward trends are significant
            short_mask = position_directions == -1
            if np.any(short_mask):
                short_prices = prices[short_mask]
                if len(short_prices) > 10:
                    # Calculate downward trend strength for shorts
                    short_trend = np.polyfit(range(len(short_prices)), short_prices, 1)[0]
                    short_trend_significance = min(abs(short_trend) / np.mean(short_prices) * 100, 1.0)
                    significance_factors.append(short_trend_significance)

            # Position-aware win rate significance
            win_rate_analysis = self.calculate_position_aware_win_rate(returns, position_directions)
            win_rate_significance = min(win_rate_analysis['overall_win_rate'] * 2, 1.0)
            significance_factors.append(win_rate_significance)

            # Position balance significance (diversification benefit)
            long_count = np.sum(position_directions == 1)
            short_count = np.sum(position_directions == -1)
            total_positions = long_count + short_count

            if total_positions > 0:
                position_balance = 1.0 - abs(long_count - short_count) / total_positions
                balance_significance = position_balance * 0.8  # Max 0.8 for perfect balance
                significance_factors.append(balance_significance)

            # Average significance
            avg_significance = np.mean(significance_factors) if significance_factors else 0.5

            return min(avg_significance, 1.0)

        except Exception as e:
            self.logger.warning(f"Regime economic significance calculation failed: {e}")
            return 0.5

    def _analyze_position_balance(
        self,
        position_directions: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze position balance and diversification benefits.

        Args:
            position_directions: Array of position directions
            returns: Array of returns

        Returns:
            Dict with position balance analysis
        """
        try:
            analysis = {
                'position_balance_score': 0.0,
                'diversification_benefit': 0.0,
                'long_short_correlation': 0.0,
                'position_stability': 0.0
            }

            long_count = np.sum(position_directions == 1)
            short_count = np.sum(position_directions == -1)
            total_positions = long_count + short_count

            if total_positions == 0:
                return analysis

            # Position balance score (closer to 50/50 is better)
            long_ratio = long_count / total_positions
            short_ratio = short_count / total_positions
            analysis['position_balance_score'] = 1.0 - abs(long_ratio - short_ratio)

            # Diversification benefit (lower correlation between long/short returns = better)
            long_returns = returns[position_directions == 1]
            short_returns = returns[position_directions == -1]

            if len(long_returns) > 1 and len(short_returns) > 1:
                # Calculate correlation between long and short returns
                combined_returns = np.zeros((min(len(long_returns), len(short_returns)), 2))
                combined_returns[:, 0] = long_returns[:len(combined_returns)]
                combined_returns[:, 1] = short_returns[:len(combined_returns)]

                correlation = np.corrcoef(combined_returns, rowvar=False)[0, 1]
                analysis['long_short_correlation'] = correlation
                analysis['diversification_benefit'] = (1.0 - abs(correlation)) * 0.5

            # Position stability (how consistent position direction is)
            position_changes = np.sum(np.abs(np.diff(position_directions))) / (len(position_directions) - 1)
            analysis['position_stability'] = 1.0 - position_changes if len(position_directions) > 1 else 1.0

            return analysis

        except Exception as e:
            self.logger.warning(f"Position balance analysis failed: {e}")
            return {
                'position_balance_score': 0.0,
                'diversification_benefit': 0.0,
                'long_short_correlation': 0.0,
                'position_stability': 0.0
            }

    def calculate_position_aware_trading_viability(
        self,
        market_data: pd.DataFrame,
        regime_predictions: np.ndarray,
        position_directions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate position-aware trading viability for both TAS and NAS systems.

        Args:
            market_data: Market data DataFrame
            regime_predictions: Regime predictions
            position_directions: Optional position directions

        Returns:
            Dict with position-aware trading viability analysis
        """
        try:
            if position_directions is None:
                # Infer positions from regime characteristics
                position_directions = self._infer_positions_from_regimes(
                    market_data, regime_predictions
                )

            # Get position-aware analysis
            position_analysis = self.analyze_regime_position_performance(
                market_data, regime_predictions, position_directions
            )

            # Calculate viability scores for long and short positions
            long_viability = self._calculate_position_viability(
                position_analysis, 'long'
            )
            short_viability = self._calculate_position_viability(
                position_analysis, 'short'
            )

            # Overall viability considering both positions
            overall_viability = self._calculate_overall_viability(
                long_viability, short_viability, position_analysis
            )

            return {
                'long_viability': long_viability,
                'short_viability': short_viability,
                'overall_viability': overall_viability,
                'position_analysis': position_analysis,
                'viability_metadata': {
                    'calculation_method': 'position_aware',
                    'minimum_profit_threshold': self.config.minimum_profit_threshold,
                    'transaction_cost': self.config.transaction_cost,
                    'supports_long_short': True
                }
            }

        except Exception as e:
            self.logger.error(f"Position-aware trading viability calculation failed: {e}")
            return {
                'long_viability': 0.5,
                'short_viability': 0.5,
                'overall_viability': 0.5,
                'position_analysis': {},
                'viability_metadata': {'error': str(e)}
            }

    def _infer_positions_from_regimes(
        self,
        market_data: pd.DataFrame,
        regime_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Infer position directions from regime characteristics.

        Args:
            market_data: Market data DataFrame
            regime_predictions: Regime predictions

        Returns:
            Array of inferred position directions
        """
        try:
            returns = market_data['close'].pct_change().values
            position_directions = np.zeros(len(market_data))

            unique_regimes = np.unique(regime_predictions)

            for regime in unique_regimes:
                regime_mask = regime_predictions == regime
                regime_returns = returns[regime_mask]

                if len(regime_returns) < 10:
                    continue

                # Calculate regime characteristics
                regime_volatility = np.std(regime_returns)
                regime_trend = np.mean(regime_returns)

                # Infer positions based on regime characteristics
                if regime_trend > 0 and regime_volatility < 0.02:
                    # Trending upward with low volatility -> long bias
                    position_directions[regime_mask] = 1
                elif regime_trend < 0 and regime_volatility < 0.02:
                    # Trending downward with low volatility -> short bias
                    position_directions[regime_mask] = -1
                else:
                    # High volatility or no clear trend -> neutral
                    position_directions[regime_mask] = 0

            return position_directions

        except Exception as e:
            self.logger.warning(f"Position inference failed: {e}")
            # Default to neutral positions
            return np.zeros(len(market_data))

    def _calculate_position_viability(
        self,
        position_analysis: Dict[str, Any],
        position_type: str
    ) -> float:
        """
        Calculate viability score for a specific position type.

        Args:
            position_analysis: Position analysis results
            position_type: 'long' or 'short'

        Returns:
            Viability score for the position type
        """
        try:
            if position_type == 'long':
                win_rate = position_analysis['overall_analysis'].get('long_win_rate', 0.5)
                trade_count = position_analysis['overall_analysis'].get('long_trades', 0)
            else:  # short
                win_rate = position_analysis['overall_analysis'].get('short_win_rate', 0.5)
                trade_count = position_analysis['overall_analysis'].get('short_trades', 0)

            # Base viability on win rate
            viability = win_rate * 0.7  # 70% weight on win rate

            # Bonus for sufficient trade count
            if trade_count > 50:
                viability += 0.1
            elif trade_count > 20:
                viability += 0.05

            return min(viability, 1.0)

        except Exception as e:
            self.logger.warning(f"Position viability calculation failed for {position_type}: {e}")
            return 0.5

    def _calculate_overall_viability(
        self,
        long_viability: float,
        short_viability: float,
        position_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate overall viability considering both long and short positions.

        Args:
            long_viability: Long position viability score
            short_viability: Short position viability score
            position_analysis: Position analysis results

        Returns:
            Overall viability score
        """
        try:
            # Weight by position balance and individual viabilities
            position_balance = position_analysis['position_balance_analysis'].get('position_balance_score', 0.5)

            # Overall viability is weighted average of long/short viabilities
            # with bonus for good position balance
            long_weight = 0.5 + (position_balance * 0.2)  # 0.5 to 0.7
            short_weight = 0.5 - (position_balance * 0.2)  # 0.5 to 0.3

            overall_viability = (
                long_viability * long_weight +
                short_viability * short_weight
            )

            # Bonus for diversification benefit
            diversification_benefit = position_analysis['position_balance_analysis'].get('diversification_benefit', 0.0)
            overall_viability += diversification_benefit * 0.1

            return min(overall_viability, 1.0)

        except Exception as e:
            self.logger.warning(f"Overall viability calculation failed: {e}")
            return 0.5

    def get_position_aware_recommendations(
        self,
        position_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get position-aware trading recommendations.

        Args:
            position_analysis: Position analysis results

        Returns:
            Dict with trading recommendations for long/short positions
        """
        try:
            recommendations = {
                'position_recommendations': {},
                'risk_management': {},
                'strategy_adjustments': {},
                'regime_priorities': {}
            }

            overall_win_rate = position_analysis['overall_analysis'].get('overall_win_rate', 0.5)
            long_win_rate = position_analysis['overall_analysis'].get('long_win_rate', 0.5)
            short_win_rate = position_analysis['overall_analysis'].get('short_win_rate', 0.5)

            # Position recommendations
            if long_win_rate >= 0.6:
                recommendations['position_recommendations']['long'] = {
                    'recommendation': 'Strong Long Position',
                    'confidence': long_win_rate,
                    'position_size': 'Full' if long_win_rate >= 0.7 else 'Reduced'
                }
            elif long_win_rate >= 0.4:
                recommendations['position_recommendations']['long'] = {
                    'recommendation': 'Cautious Long Position',
                    'confidence': long_win_rate,
                    'position_size': 'Small'
                }
            else:
                recommendations['position_recommendations']['long'] = {
                    'recommendation': 'Avoid Long Positions',
                    'confidence': 1 - long_win_rate,
                    'position_size': 'None'
                }

            if short_win_rate >= 0.6:
                recommendations['position_recommendations']['short'] = {
                    'recommendation': 'Strong Short Position',
                    'confidence': short_win_rate,
                    'position_size': 'Full' if short_win_rate >= 0.7 else 'Reduced'
                }
            elif short_win_rate >= 0.4:
                recommendations['position_recommendations']['short'] = {
                    'recommendation': 'Cautious Short Position',
                    'confidence': short_win_rate,
                    'position_size': 'Small'
                }
            else:
                recommendations['position_recommendations']['short'] = {
                    'recommendation': 'Avoid Short Positions',
                    'confidence': 1 - short_win_rate,
                    'position_size': 'None'
                }

            # Risk management recommendations
            position_balance = position_analysis['position_balance_analysis'].get('position_balance_score', 0.5)
            if position_balance > 0.7:
                recommendations['risk_management']['position_balance'] = 'Good diversification'
            elif position_balance > 0.4:
                recommendations['risk_management']['position_balance'] = 'Moderate diversification'
            else:
                recommendations['risk_management']['position_balance'] = 'Poor diversification - consider balancing'

            return recommendations

        except Exception as e:
            self.logger.warning(f"Position-aware recommendations failed: {e}")
            return {}


def create_position_aware_analyzer(config: PositionAwareConfig = None) -> PositionAwareTradingAnalyzer:
    """
    Create a position-aware trading analyzer instance.

    Args:
        config: Position-aware configuration

    Returns:
        PositionAwareTradingAnalyzer instance
    """
    return PositionAwareTradingAnalyzer(config)


def quick_position_aware_analysis(
    market_data: pd.DataFrame,
    regime_predictions: np.ndarray,
    position_directions: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Quick position-aware analysis with default settings.

    Args:
        market_data: Market data DataFrame
        regime_predictions: Regime predictions
        position_directions: Optional position directions

    Returns:
        Dict with position-aware analysis results
    """
    analyzer = PositionAwareTradingAnalyzer()
    return analyzer.analyze_regime_position_performance(
        market_data, regime_predictions, position_directions
    )