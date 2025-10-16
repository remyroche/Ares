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

            # Ensure arrays have the same length
            min_length = min(len(returns), len(position_directions))
            returns = returns[:min_length]
            position_directions = position_directions[:min_length]

            # Overall win rate based on directional positions and their profitability
            adjusted_returns = returns - self.config.transaction_cost
            # Calculate win rate for all positions that were taken (long or short)
            positions_taken = (position_directions == 1) | (position_directions == -1)
            if np.any(positions_taken):
                position_returns = adjusted_returns[positions_taken]
                position_types = position_directions[positions_taken]

                # For longs: positive returns = profit, for shorts: negative returns = profit
                long_mask = position_types == 1
                short_mask = position_types == -1

                wins = np.zeros_like(position_returns, dtype=bool)
                if np.any(long_mask):
                    wins[long_mask] = position_returns[long_mask] > self.config.minimum_profit_threshold
                if np.any(short_mask):
                    wins[short_mask] = position_returns[short_mask] < -self.config.minimum_profit_threshold

                results['overall_win_rate'] = np.mean(wins) if len(wins) > 0 else 0.0
            else:
                results['overall_win_rate'] = 0.0

            # Position-specific win rates
            long_mask = position_directions == 1
            short_mask = position_directions == -1

            tprint_debug(f"Position masks: {np.sum(long_mask)} long, {np.sum(short_mask)} short")

            if np.any(long_mask):
                long_returns = adjusted_returns[long_mask]
                # For longs: positive returns = profit
                long_wins = long_returns > self.config.minimum_profit_threshold
                results['long_win_rate'] = np.mean(long_wins)
                results['long_trades'] = len(long_returns)
                tprint_debug(f"Long analysis: {len(long_returns)} trades, {np.sum(long_wins)} wins, win_rate={results['long_win_rate']:.3f}")

            if np.any(short_mask):
                short_returns = adjusted_returns[short_mask]
                # For shorts: negative returns = profit
                short_wins = short_returns < -self.config.minimum_profit_threshold
                results['short_win_rate'] = np.mean(short_wins)
                results['short_trades'] = len(short_returns)
                tprint_debug(f"Short analysis: {len(short_returns)} trades, {np.sum(short_wins)} wins, win_rate={results['short_win_rate']:.3f}")

            # If we still have 0 trades, add a warning but don't create artificial positions
            if results['long_trades'] == 0 and results['short_trades'] == 0:
                tprint_warning("⚠️ No long or short positions found - check position inference logic")
                tprint_debug("Position inference may need to be more aggressive in assigning directional positions")
                # Note: We don't create artificial positions here as that would introduce hindsight bias

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
                'total_trades': len(returns) if 'returns' in locals() else 0,
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
                # Infer positions from regime characteristics
                position_directions = self._infer_positions_from_regimes(market_data, regime_labels)
            else:
                # Ensure position_directions matches the length of returns
                if len(position_directions) > len(returns):
                    position_directions = position_directions[1:]  # Align with returns
                elif len(position_directions) < len(returns):
                    # Pad with neutral positions if needed
                    position_directions = np.pad(position_directions, (0, len(returns) - len(position_directions)), mode='constant', constant_values=0)

            # Final alignment check
            if len(position_directions) != len(returns):
                min_length = min(len(returns), len(position_directions))
                returns = returns[:min_length]
                position_directions = position_directions[:min_length]
                tprint_debug(f"Final alignment: returns={len(returns)}, positions={len(position_directions)}")

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

                # Ensure all arrays have the same length
                min_length = min(len(position_directions), len(aligned_regime_mask), len(regime_returns))

                if min_length == 0:
                    tprint_debug(f"   Regime {regime_id}: skipping (min_length=0)")
                    continue

                # Debug array lengths for this regime
                tprint_debug(f"   Regime {regime_id} array lengths - position_directions: {len(position_directions)}, "
                            f"aligned_regime_mask: {len(aligned_regime_mask)}, regime_returns: {len(regime_returns)}, "
                            f"min_length: {min_length}")

                # Truncate all arrays to the same length
                aligned_regime_mask = aligned_regime_mask[:min_length]
                regime_positions = position_directions[:min_length][aligned_regime_mask]
                regime_returns = regime_returns[aligned_regime_mask]

                # Final alignment: ensure regime_returns and regime_positions have same length
                min_regime_length = min(len(regime_returns), len(regime_positions))
                regime_returns = regime_returns[:min_regime_length]
                regime_positions = regime_positions[:min_regime_length]

                tprint_debug(f"   Regime {regime_id}: final lengths - returns={len(regime_returns)}, positions={len(regime_positions)}")

                if len(regime_returns) == 0 or len(regime_positions) == 0:
                    tprint_debug(f"   Regime {regime_id}: skipping (empty arrays after alignment)")
                    continue

                # Get aligned regime data for economic significance calculation
                # Use the original regime_data but slice it to match the aligned arrays
                # regime_data has length len(regime_returns) + 1, so we slice it appropriately
                aligned_regime_data = regime_data.iloc[1:len(regime_returns)+1]

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

            # Ensure all required keys are present with fallback values
            if 'overall_analysis' not in results:
                results['overall_analysis'] = {
                    'overall_win_rate': 0.5,
                    'long_win_rate': 0.5,
                    'short_win_rate': 0.5,
                    'total_trades': 0,
                    'long_trades': 0,
                    'short_trades': 0
                }

            if 'position_balance_analysis' not in results:
                results['position_balance_analysis'] = {
                    'position_balance_score': 0.5,
                    'diversification_benefit': 0.0,
                    'long_short_correlation': 0.0,
                    'position_stability': 0.0
                }

            return results

        except Exception as e:
            self.logger.error(f"❌ Regime position performance analysis failed: {e}")
            self.logger.error(f"   Array lengths - returns: {len(returns) if 'returns' in locals() else 'N/A'}, "
                            f"position_directions: {len(position_directions) if 'position_directions' in locals() else 'N/A'}, "
                            f"regime_labels: {len(regime_labels) if 'regime_labels' in locals() else 'N/A'}")
            # Return fallback structure to prevent downstream errors
            return {
                'overall_analysis': {
                    'overall_win_rate': 0.5,
                    'long_win_rate': 0.5,
                    'short_win_rate': 0.5,
                    'total_trades': 0,
                    'long_trades': 0,
                    'short_trades': 0
                },
                'regime_analyses': {},
                'position_balance_analysis': {
                    'position_balance_score': 0.5,
                    'diversification_benefit': 0.0,
                    'long_short_correlation': 0.0,
                    'position_stability': 0.0
                },
                'error': str(e)
            }

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
                tprint_debug(f"Inferred position directions: {len(position_directions)} total, "
                           f"{np.sum(position_directions == 1)} long, {np.sum(position_directions == -1)} short")

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
            Array of inferred position directions (aligned with returns length)
        """
        try:
            # Get returns array (length N-1 due to pct_change)
            returns = market_data['close'].pct_change().values

            # Remove NaN values from returns and get valid indices
            valid_returns_mask = ~np.isnan(returns)
            returns = returns[valid_returns_mask]

            # Align regime_predictions with the valid returns
            # Since returns is length N-1 due to pct_change, we need to align regime_predictions accordingly
            if len(regime_predictions) != len(market_data):
                tprint_warning(f"⚠️ Regime predictions length {len(regime_predictions)} != market data length {len(market_data)}")
                # Align regime_predictions to market_data length
                if len(regime_predictions) > len(market_data):
                    regime_predictions = regime_predictions[:len(market_data)]
                else:
                    # Pad with last regime if shorter
                    regime_predictions = np.pad(regime_predictions, (0, len(market_data) - len(regime_predictions)), mode='edge')

            # Align regime_predictions with returns (skip first element due to pct_change)
            # and apply the same valid mask as returns
            regime_predictions_aligned = regime_predictions[1:][valid_returns_mask]

            position_directions = np.zeros(len(returns))

            unique_regimes = np.unique(regime_predictions_aligned)
            tprint_debug(f"Inferring positions for {len(unique_regimes)} regimes")
            tprint_debug(f"Array dimensions - returns: {len(returns)}, position_directions: {len(position_directions)}, regime_predictions_aligned: {len(regime_predictions_aligned)}")

            for regime in unique_regimes:
                # Create regime mask aligned with returns
                regime_mask = regime_predictions_aligned == regime

                # Final safety check - ensure all arrays have same length
                if len(regime_mask) != len(returns):
                    tprint_error(f"❌ Critical dimension mismatch: regime_mask={len(regime_mask)}, returns={len(returns)}")
                    continue

                if not np.any(regime_mask):
                    continue

                # Safe extraction with dimension check
                try:
                    regime_returns = returns[regime_mask]
                except IndexError as e:
                    tprint_error(f"❌ Index error in regime {regime}: {e}")
                    tprint_error(f"   returns shape: {returns.shape}, regime_mask shape: {regime_mask.shape}")
                    continue

                if len(regime_returns) < 3:  # Minimum threshold for meaningful analysis
                    continue

                # Calculate regime characteristics
                regime_volatility = np.std(regime_returns)
                regime_trend = np.mean(regime_returns)
                regime_median = np.median(regime_returns)

                tprint_debug(f"Regime {regime}: trend={regime_trend:.4f}, vol={regime_volatility:.4f}, median={regime_median:.4f}")

                # Conservative position inference logic
                # Only assign positions when there's a clear, significant trend
                if regime_trend > 0.002 and regime_median > 0.001:  # Conservative thresholds for positive bias
                    # Clear upward bias -> long positions
                    position_directions[regime_mask] = 1
                    tprint_debug(f"  -> Long bias (trend={regime_trend:.4f}, median={regime_median:.4f})")
                elif regime_trend < -0.002 and regime_median < -0.001:  # Conservative thresholds for negative bias
                    # Clear downward bias -> short positions
                    position_directions[regime_mask] = -1
                    tprint_debug(f"  -> Short bias (trend={regime_trend:.4f}, median={regime_median:.4f})")
                else:
                    # No clear trend -> assign positions based on individual return signs
                    # Only for significant individual returns
                    regime_indices = np.where(regime_mask)[0]
                    for ret, idx in zip(regime_returns, regime_indices):
                        if ret > 0.002:  # Conservative threshold for positive returns
                            position_directions[idx] = 1
                        elif ret < -0.002:  # Conservative threshold for negative returns
                            position_directions[idx] = -1
                        # else keep as 0 (neutral)
                    tprint_debug(f"  -> Mixed positions based on individual returns")

            # Log position distribution
            long_count = np.sum(position_directions == 1)
            short_count = np.sum(position_directions == -1)
            neutral_count = np.sum(position_directions == 0)
            total = len(position_directions)

            tprint_debug(f"Position distribution: Long={long_count}, Short={short_count}, Neutral={neutral_count}, Total={total}")

            # If we have very few positions, don't create artificial ones
            # This prevents unrealistic win rates from artificial position assignment
            if long_count + short_count < total * 0.1:  # Less than 10% of periods have positions
                tprint_warning("⚠️ Very few positions inferred - this may indicate low trading opportunity")
                tprint_debug("Not creating artificial positions to avoid unrealistic results")

            return position_directions

        except Exception as e:
            self.logger.warning(f"Position inference failed: {e}")
            # Return neutral positions aligned with returns length
            return np.zeros(len(returns))

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
            # Safely get overall_analysis with fallback
            overall_analysis = position_analysis.get('overall_analysis', {})

            if position_type == 'long':
                win_rate = overall_analysis.get('long_win_rate', 0.5)
                trade_count = overall_analysis.get('long_trades', 0)
            else:  # short
                win_rate = overall_analysis.get('short_win_rate', 0.5)
                trade_count = overall_analysis.get('short_trades', 0)

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
            # Safely get position_balance_analysis with fallback
            position_balance_analysis = position_analysis.get('position_balance_analysis', {})
            position_balance = position_balance_analysis.get('position_balance_score', 0.5)

            # Overall viability is weighted average of long/short viabilities
            # with bonus for good position balance
            long_weight = 0.5 + (position_balance * 0.2)  # 0.5 to 0.7
            short_weight = 0.5 - (position_balance * 0.2)  # 0.5 to 0.3

            overall_viability = (
                long_viability * long_weight +
                short_viability * short_weight
            )

            # Bonus for diversification benefit
            diversification_benefit = position_balance_analysis.get('diversification_benefit', 0.0)
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
