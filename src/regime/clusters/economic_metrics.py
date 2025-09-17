"""
Economic Metrics for Regime Quality Validation.

This module provides comprehensive economic metrics to validate whether
discovered regimes represent economically meaningful market behaviors/patterns
that are relevant for ML model training and trading.

Key Economic Validation Questions:
1. Do regimes have different risk-return profiles?
2. Are regimes economically persistent enough for ML training?
3. Do regimes show different market microstructure behaviors?
4. Can regimes be exploited for trading (economic significance)?
5. Do regimes have different drawdown characteristics?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

from src.utils.logger import system_logger


class EconomicMetric(Enum):
    """Economic metrics for regime validation."""
    # Risk-Return Metrics
    SHARPE_RATIO_DIFFERENCE = "sharpe_ratio_difference"
    INFORMATION_RATIO_DIFFERENCE = "information_ratio_difference"
    RETURN_SEPARABILITY = "return_separability"
    RISK_ADJUSTED_RETURN_DIFF = "risk_adjusted_return_diff"
    
    # Drawdown and Risk Metrics
    MAXIMUM_DRAWDOWN_DIFFERENCE = "maximum_drawdown_difference"
    VALUE_AT_RISK_DIFFERENCE = "value_at_risk_difference"
    EXPECTED_SHORTFALL_DIFFERENCE = "expected_shortfall_difference"
    VOLATILITY_REGIME_SIGNIFICANCE = "volatility_regime_significance"
    
    # Market Microstructure Economics
    VOLUME_PROFILE_DIFFERENCE = "volume_profile_difference"
    MARKET_IMPACT_DIFFERENCE = "market_impact_difference"
    
    # Price Action Influence Metrics
    PRICE_ACTION_INFLUENCE = "price_action_influence"
    MOMENTUM_PREDICTION_POWER = "momentum_prediction_power"
    MEAN_REVERSION_SIGNAL_STRENGTH = "mean_reversion_signal_strength"
    
    # Trading Economics
    REGIME_PERSISTENCE_VALUE = "regime_persistence_value"
    TRANSITION_COST_ANALYSIS = "transition_cost_analysis"
    REGIME_PREDICTION_VALUE = "regime_prediction_value"
    ECONOMIC_REGIME_STABILITY = "economic_regime_stability"


@dataclass
class EconomicValidationConfig:
    """Configuration for economic validation."""
    # Risk-free rate for Sharpe ratio calculations
    risk_free_rate: float = 0.02  # 2% annual
    
    # Trading cost assumptions
    transaction_cost: float = 0.001  # 0.1% per trade
    market_impact_cost: float = 0.0005  # 0.05% market impact
    
    # Risk metrics parameters
    confidence_level: float = 0.05  # 95% VaR/ES
    lookback_periods: List[int] = None  # [21, 63, 252] trading days
    
    # Persistence thresholds
    min_regime_persistence_days: int = 10
    economic_significance_threshold: float = 0.01  # 1% annual return difference
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [21, 63, 252]  # 1M, 3M, 1Y


@dataclass
class EconomicValidationResult:
    """Result container for economic validation."""
    metric: EconomicMetric
    value: float
    economic_significance: bool
    regime_specific_values: Dict[int, float]
    statistical_significance: Optional[float]  # p-value
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    trading_implications: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric.value,
            'value': self.value,
            'economic_significance': self.economic_significance,
            'regime_specific_values': self.regime_specific_values,
            'statistical_significance': self.statistical_significance,
            'confidence_interval': self.confidence_interval,
            'interpretation': self.interpretation,
            'trading_implications': self.trading_implications,
            'metadata': self.metadata
        }


class EconomicValidator:
    """Main class for economic validation of regime quality."""
    
    def __init__(self, config: Optional[EconomicValidationConfig] = None):
        self.config = config or EconomicValidationConfig()
        self.logger = system_logger.getChild('EconomicValidator')
    
    def validate_regime_economics(self, 
                                market_data: pd.DataFrame,
                                regime_labels: np.ndarray) -> Dict[EconomicMetric, EconomicValidationResult]:
        """
        Comprehensive economic validation of discovered regimes.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Regime assignments
            
        Returns:
            Dictionary mapping economic metrics to validation results
        """
        self.logger.info("💰 Starting comprehensive economic validation of regimes")
        
        results = {}
        
        # 1. Risk-Return Profile Analysis
        results.update(self._validate_risk_return_profiles(market_data, regime_labels))
        
        # 2. Drawdown and Risk Analysis
        results.update(self._validate_drawdown_characteristics(market_data, regime_labels))
        
        # 3. Market Microstructure Economics
        results.update(self._validate_microstructure_economics(market_data, regime_labels))
        
        # 4. Trading Economics
        results.update(self._validate_trading_economics(market_data, regime_labels))
        
        self.logger.info(f"✅ Completed economic validation: {len(results)} metrics")
        return results
    
    def _validate_risk_return_profiles(self, 
                                     market_data: pd.DataFrame,
                                     regime_labels: np.ndarray) -> Dict[EconomicMetric, EconomicValidationResult]:
        """Validate risk-return profiles across regimes."""
        results = {}
        
        # Calculate returns
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().dropna()
            regime_labels_aligned = regime_labels[1:]  # Align with returns
        else:
            self.logger.warning("No close price data for risk-return analysis")
            return results
        
        # Group returns by regime
        unique_regimes = np.unique(regime_labels_aligned)
        regime_returns = {}
        
        for regime in unique_regimes:
            mask = regime_labels_aligned == regime
            regime_returns[regime] = returns[mask].dropna()
        
        # 1. Sharpe Ratio Differences
        results[EconomicMetric.SHARPE_RATIO_DIFFERENCE] = self._calculate_sharpe_ratio_difference(regime_returns)
        
        # 2. Information Ratio Differences
        results[EconomicMetric.INFORMATION_RATIO_DIFFERENCE] = self._calculate_information_ratio_difference(regime_returns)
        
        # 3. Return Separability
        results[EconomicMetric.RETURN_SEPARABILITY] = self._calculate_return_separability(regime_returns)
        
        # 4. Risk-Adjusted Return Differences
        results[EconomicMetric.RISK_ADJUSTED_RETURN_DIFF] = self._calculate_risk_adjusted_return_diff(regime_returns)
        
        return results
    
    def _calculate_sharpe_ratio_difference(self, regime_returns: Dict[int, pd.Series]) -> EconomicValidationResult:
        """Calculate Sharpe ratio differences between regimes."""
        regime_sharpe_ratios = {}
        
        for regime, returns in regime_returns.items():
            if len(returns) > 0:
                excess_returns = returns - self.config.risk_free_rate / 252  # Daily risk-free rate
                sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                regime_sharpe_ratios[regime] = float(sharpe_ratio)
            else:
                regime_sharpe_ratios[regime] = 0.0
        
        # Calculate maximum difference
        if len(regime_sharpe_ratios) > 1:
            sharpe_values = list(regime_sharpe_ratios.values())
            max_diff = max(sharpe_values) - min(sharpe_values)
            
            # Economic significance: >0.5 Sharpe ratio difference is meaningful
            economically_significant = max_diff > 0.5
            
            # Statistical significance (t-test between best and worst regimes)
            best_regime = max(regime_sharpe_ratios, key=regime_sharpe_ratios.get)
            worst_regime = min(regime_sharpe_ratios, key=regime_sharpe_ratios.get)
            
            best_returns = regime_returns[best_regime]
            worst_returns = regime_returns[worst_regime]
            
            if len(best_returns) > 10 and len(worst_returns) > 10:
                t_stat, p_value = stats.ttest_ind(best_returns, worst_returns)
                p_value = float(p_value)
            else:
                p_value = None
            
            interpretation = f"Maximum Sharpe ratio difference: {max_diff:.3f}"
            if economically_significant:
                interpretation += " (Economically significant)"
            
            trading_implications = "High Sharpe ratio differences suggest regimes suitable for regime-specific strategies" if economically_significant else "Limited regime-specific strategy potential"
            
        else:
            max_diff = 0.0
            economically_significant = False
            p_value = None
            interpretation = "Insufficient regimes for comparison"
            trading_implications = "Single regime - no regime-specific strategy potential"
        
        return EconomicValidationResult(
            metric=EconomicMetric.SHARPE_RATIO_DIFFERENCE,
            value=max_diff,
            economic_significance=economically_significant,
            regime_specific_values=regime_sharpe_ratios,
            statistical_significance=p_value,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'risk_free_rate': self.config.risk_free_rate}
        )
    
    def _calculate_information_ratio_difference(self, regime_returns: Dict[int, pd.Series]) -> EconomicValidationResult:
        """Calculate Information Ratio differences between regimes."""
        # Use overall market return as benchmark
        all_returns = pd.concat(regime_returns.values())
        benchmark_return = all_returns.mean()
        
        regime_info_ratios = {}
        
        for regime, returns in regime_returns.items():
            if len(returns) > 0:
                excess_returns = returns - benchmark_return
                tracking_error = excess_returns.std()
                info_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0
                regime_info_ratios[regime] = float(info_ratio * np.sqrt(252))  # Annualized
            else:
                regime_info_ratios[regime] = 0.0
        
        # Calculate maximum difference
        if len(regime_info_ratios) > 1:
            info_values = list(regime_info_ratios.values())
            max_diff = max(info_values) - min(info_values)
            
            # Economic significance: >0.3 Information Ratio difference
            economically_significant = max_diff > 0.3
            
            interpretation = f"Maximum Information Ratio difference: {max_diff:.3f}"
            trading_implications = "High IR differences indicate strong regime-specific alpha potential" if economically_significant else "Limited alpha generation potential from regimes"
        else:
            max_diff = 0.0
            economically_significant = False
            interpretation = "Insufficient regimes for comparison"
            trading_implications = "Single regime - no alpha generation potential"
        
        return EconomicValidationResult(
            metric=EconomicMetric.INFORMATION_RATIO_DIFFERENCE,
            value=max_diff,
            economic_significance=economically_significant,
            regime_specific_values=regime_info_ratios,
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'benchmark_return': float(benchmark_return)}
        )
    
    def _calculate_return_separability(self, regime_returns: Dict[int, pd.Series]) -> EconomicValidationResult:
        """Calculate economic separability of returns between regimes."""
        regime_means = {regime: float(returns.mean() * 252) for regime, returns in regime_returns.items()}  # Annualized
        regime_stds = {regime: float(returns.std() * np.sqrt(252)) for regime, returns in regime_returns.items()}  # Annualized
        
        if len(regime_means) > 1:
            # Calculate range of returns
            return_range = max(regime_means.values()) - min(regime_means.values())
            
            # Economic significance: >1% annual return difference
            economically_significant = return_range > self.config.economic_significance_threshold
            
            # Statistical test (ANOVA)
            if len(regime_returns) > 1:
                regime_return_lists = [returns.values for returns in regime_returns.values() if len(returns) > 0]
                if len(regime_return_lists) > 1:
                    f_stat, p_value = stats.f_oneway(*regime_return_lists)
                    p_value = float(p_value)
                else:
                    p_value = None
            else:
                p_value = None
            
            interpretation = f"Return separability: {return_range:.1%} annual range"
            trading_implications = f"Return differences {'justify' if economically_significant else 'may not justify'} regime-specific ML models"
        else:
            return_range = 0.0
            economically_significant = False
            p_value = None
            interpretation = "Single regime - no return separability"
            trading_implications = "No economic basis for regime-specific models"
        
        return EconomicValidationResult(
            metric=EconomicMetric.RETURN_SEPARABILITY,
            value=return_range,
            economic_significance=economically_significant,
            regime_specific_values=regime_means,
            statistical_significance=p_value,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_stds': regime_stds, 'threshold': self.config.economic_significance_threshold}
        )
    
    def _calculate_risk_adjusted_return_diff(self, regime_returns: Dict[int, pd.Series]) -> EconomicValidationResult:
        """Calculate risk-adjusted return differences (Return/Volatility)."""
        regime_risk_adj_returns = {}
        
        for regime, returns in regime_returns.items():
            if len(returns) > 0 and returns.std() > 0:
                risk_adj_return = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
                regime_risk_adj_returns[regime] = float(risk_adj_return)
            else:
                regime_risk_adj_returns[regime] = 0.0
        
        if len(regime_risk_adj_returns) > 1:
            values = list(regime_risk_adj_returns.values())
            max_diff = max(values) - min(values)
            
            # Economic significance: >0.5 risk-adjusted return difference
            economically_significant = max_diff > 0.5
            
            interpretation = f"Risk-adjusted return difference: {max_diff:.3f}"
            trading_implications = "Significant risk-adjusted differences support regime-specific risk management" if economically_significant else "Limited risk management benefits from regimes"
        else:
            max_diff = 0.0
            economically_significant = False
            interpretation = "Single regime - no risk-adjusted comparison"
            trading_implications = "No risk management benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.RISK_ADJUSTED_RETURN_DIFF,
            value=max_diff,
            economic_significance=economically_significant,
            regime_specific_values=regime_risk_adj_returns,
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={}
        )
    
    def _validate_drawdown_characteristics(self, 
                                         market_data: pd.DataFrame,
                                         regime_labels: np.ndarray) -> Dict[EconomicMetric, EconomicValidationResult]:
        """Validate drawdown and risk characteristics across regimes."""
        results = {}
        
        if 'close' not in market_data.columns:
            return results
        
        # Calculate cumulative returns and drawdowns
        returns = market_data['close'].pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Align with regime labels
        regime_labels_aligned = regime_labels[:len(drawdowns)]
        
        # Group drawdowns by regime
        unique_regimes = np.unique(regime_labels_aligned)
        regime_drawdowns = {}
        
        for regime in unique_regimes:
            mask = regime_labels_aligned == regime
            regime_drawdowns[regime] = drawdowns[mask]
        
        # Calculate maximum drawdown differences
        results[EconomicMetric.MAXIMUM_DRAWDOWN_DIFFERENCE] = self._calculate_max_drawdown_difference(regime_drawdowns)
        
        # Calculate VaR differences
        results[EconomicMetric.VALUE_AT_RISK_DIFFERENCE] = self._calculate_var_difference(regime_labels_aligned, returns)
        
        return results
    
    def _calculate_max_drawdown_difference(self, regime_drawdowns: Dict[int, pd.Series]) -> EconomicValidationResult:
        """Calculate maximum drawdown differences between regimes."""
        regime_max_drawdowns = {}
        
        for regime, drawdowns in regime_drawdowns.items():
            if len(drawdowns) > 0:
                max_dd = float(drawdowns.min())  # Most negative value
                regime_max_drawdowns[regime] = max_dd
            else:
                regime_max_drawdowns[regime] = 0.0
        
        if len(regime_max_drawdowns) > 1:
            dd_values = list(regime_max_drawdowns.values())
            # Difference between worst and best drawdown
            max_diff = abs(min(dd_values) - max(dd_values))  # Both are negative, so this gives positive difference
            
            # Economic significance: >5% drawdown difference
            economically_significant = max_diff > 0.05
            
            interpretation = f"Maximum drawdown difference: {max_diff:.1%}"
            trading_implications = "Significant drawdown differences suggest regime-specific risk controls needed" if economically_significant else "Similar risk profiles across regimes"
        else:
            max_diff = 0.0
            economically_significant = False
            interpretation = "Single regime - no drawdown comparison"
            trading_implications = "No regime-specific risk control benefits"
        
        return EconomicValidationResult(
            metric=EconomicMetric.MAXIMUM_DRAWDOWN_DIFFERENCE,
            value=max_diff,
            economic_significance=economically_significant,
            regime_specific_values=regime_max_drawdowns,
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={}
        )
    
    def _calculate_var_difference(self, regime_labels: np.ndarray, returns: pd.Series) -> EconomicValidationResult:
        """Calculate Value at Risk differences between regimes."""
        unique_regimes = np.unique(regime_labels)
        regime_vars = {}
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) > 0:
                var_95 = float(np.percentile(regime_returns, self.config.confidence_level * 100))
                regime_vars[regime] = var_95
            else:
                regime_vars[regime] = 0.0
        
        if len(regime_vars) > 1:
            var_values = list(regime_vars.values())
            # Difference between worst and best VaR (both negative)
            max_diff = abs(min(var_values) - max(var_values))
            
            # Economic significance: >1% daily VaR difference
            economically_significant = max_diff > 0.01
            
            interpretation = f"95% VaR difference: {max_diff:.2%} daily"
            trading_implications = "Significant VaR differences require regime-specific position sizing" if economically_significant else "Similar tail risk across regimes"
        else:
            max_diff = 0.0
            economically_significant = False
            interpretation = "Single regime - no VaR comparison"
            trading_implications = "No position sizing benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.VALUE_AT_RISK_DIFFERENCE,
            value=max_diff,
            economic_significance=economically_significant,
            regime_specific_values=regime_vars,
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'confidence_level': self.config.confidence_level}
        )
    
    def _validate_microstructure_economics(self, 
                                         market_data: pd.DataFrame,
                                         regime_labels: np.ndarray) -> Dict[EconomicMetric, EconomicValidationResult]:
        """Validate market microstructure economics across regimes."""
        results = {}
        
        # Volume profile differences
        if 'volume' in market_data.columns:
            results[EconomicMetric.VOLUME_PROFILE_DIFFERENCE] = self._calculate_volume_profile_difference(market_data, regime_labels)
        
        # Price action influence metrics
        if 'close' in market_data.columns:
            results[EconomicMetric.PRICE_ACTION_INFLUENCE] = self._calculate_price_action_influence(market_data, regime_labels)
            results[EconomicMetric.MOMENTUM_PREDICTION_POWER] = self._calculate_momentum_prediction_power(market_data, regime_labels)
            results[EconomicMetric.MEAN_REVERSION_SIGNAL_STRENGTH] = self._calculate_mean_reversion_signal_strength(market_data, regime_labels)
        
        return results
    
    def _calculate_volume_profile_difference(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate volume profile differences between regimes."""
        unique_regimes = np.unique(regime_labels)
        regime_volume_profiles = {}
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_volume = market_data['volume'][mask]
            
            if len(regime_volume) > 0:
                # Volume statistics
                volume_profile = {
                    'mean_volume': float(regime_volume.mean()),
                    'median_volume': float(regime_volume.median()),
                    'volume_volatility': float(regime_volume.std())
                }
                regime_volume_profiles[regime] = volume_profile
        
        if len(regime_volume_profiles) > 1:
            # Compare mean volumes
            mean_volumes = [profile['mean_volume'] for profile in regime_volume_profiles.values()]
            volume_range = max(mean_volumes) - min(mean_volumes)
            relative_diff = volume_range / np.mean(mean_volumes) if np.mean(mean_volumes) > 0 else 0
            
            # Economic significance: >50% relative volume difference
            economically_significant = relative_diff > 0.5
            
            interpretation = f"Volume profile difference: {relative_diff:.1%} relative"
            trading_implications = "Significant volume differences suggest regime-specific execution strategies" if economically_significant else "Similar liquidity across regimes"
        else:
            relative_diff = 0.0
            economically_significant = False
            interpretation = "Single regime - no volume comparison"
            trading_implications = "No execution strategy benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.VOLUME_PROFILE_DIFFERENCE,
            value=relative_diff,
            economic_significance=economically_significant,
            regime_specific_values={regime: profile['mean_volume'] for regime, profile in regime_volume_profiles.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'full_profiles': regime_volume_profiles}
        )
    
    def _calculate_liquidity_cost_difference(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate liquidity cost differences using spread proxy."""
        # Use high-low spread as liquidity cost proxy
        spread_proxy = (market_data['high'] - market_data['low']) / market_data['close']
        
        unique_regimes = np.unique(regime_labels)
        regime_liquidity_costs = {}
        
        for regime in unique_regimes:
            mask = regime_labels == regime
            regime_spreads = spread_proxy[mask]
            
            if len(regime_spreads) > 0:
                avg_spread = float(regime_spreads.mean())
                regime_liquidity_costs[regime] = avg_spread
        
        if len(regime_liquidity_costs) > 1:
            spread_values = list(regime_liquidity_costs.values())
            max_diff = max(spread_values) - min(spread_values)
            
            # Economic significance: >0.1% spread difference
            economically_significant = max_diff > 0.001
            
            interpretation = f"Liquidity cost difference: {max_diff:.3%}"
            trading_implications = "Significant spread differences require regime-specific execution" if economically_significant else "Similar transaction costs across regimes"
        else:
            max_diff = 0.0
            economically_significant = False
            interpretation = "Single regime - no liquidity comparison"
            trading_implications = "No execution cost benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.LIQUIDITY_COST_DIFFERENCE,
            value=max_diff,
            economic_significance=economically_significant,
            regime_specific_values=regime_liquidity_costs,
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={}
        )
    
    def _validate_trading_economics(self, 
                                  market_data: pd.DataFrame,
                                  regime_labels: np.ndarray) -> Dict[EconomicMetric, EconomicValidationResult]:
        """Validate trading economics of regimes."""
        results = {}
        
        # Regime persistence value
        results[EconomicMetric.REGIME_PERSISTENCE_VALUE] = self._calculate_regime_persistence_value(regime_labels)
        
        # Transition cost analysis
        results[EconomicMetric.TRANSITION_COST_ANALYSIS] = self._calculate_transition_cost_analysis(regime_labels)
        
        return results
    
    def _calculate_regime_persistence_value(self, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate the economic value of regime persistence."""
        # Calculate regime durations
        regime_durations = []
        current_regime = regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = regime_labels[i]
                current_duration = 1
        regime_durations.append(current_duration)
        
        # Calculate persistence metrics
        avg_duration = np.mean(regime_durations)
        median_duration = np.median(regime_durations)
        min_duration = np.min(regime_durations)
        
        # Economic significance: average duration > minimum threshold
        economically_significant = avg_duration >= self.config.min_regime_persistence_days
        
        # Persistence value (higher is better for ML training)
        persistence_value = min(avg_duration / self.config.min_regime_persistence_days, 3.0)  # Cap at 3x
        
        interpretation = f"Average regime duration: {avg_duration:.1f} periods"
        if economically_significant:
            trading_implications = "Sufficient persistence for ML model training and strategy deployment"
        else:
            trading_implications = "Limited persistence may reduce ML model effectiveness"
        
        return EconomicValidationResult(
            metric=EconomicMetric.REGIME_PERSISTENCE_VALUE,
            value=float(persistence_value),
            economic_significance=economically_significant,
            regime_specific_values={'avg_duration': avg_duration, 'median_duration': median_duration, 'min_duration': min_duration},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'durations': regime_durations, 'threshold': self.config.min_regime_persistence_days}
        )
    
    def _calculate_transition_cost_analysis(self, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate the cost of regime transitions."""
        # Count transitions
        transitions = np.sum(np.diff(regime_labels) != 0)
        total_periods = len(regime_labels) - 1
        transition_frequency = transitions / total_periods if total_periods > 0 else 0
        
        # Estimate transition costs (assuming position changes on regime change)
        total_transition_cost = transitions * (self.config.transaction_cost + self.config.market_impact_cost)
        annualized_cost = total_transition_cost * (252 / total_periods) if total_periods > 0 else 0
        
        # Economic significance: <2% annual cost from transitions
        economically_significant = annualized_cost < 0.02
        
        interpretation = f"Transition frequency: {transition_frequency:.1%}, Annual cost: {annualized_cost:.2%}"
        trading_implications = "Reasonable transition costs support regime-based strategies" if economically_significant else "High transition costs may limit strategy profitability"
        
        return EconomicValidationResult(
            metric=EconomicMetric.TRANSITION_COST_ANALYSIS,
            value=float(annualized_cost),
            economic_significance=economically_significant,
            regime_specific_values={'transition_frequency': transition_frequency, 'total_transitions': int(transitions)},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'transaction_cost': self.config.transaction_cost, 'market_impact_cost': self.config.market_impact_cost}
        )
    
    def _calculate_price_action_influence(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate how much regimes influence subsequent price action."""
        
        # Calculate returns and regime-specific price action patterns
        returns = market_data['close'].pct_change().fillna(0)
        
        # Look at next-period price action given current regime
        unique_regimes = np.unique(regime_labels)
        regime_price_influence = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            # Get returns following this regime (lead by 1 period)
            following_returns = []
            for i in range(len(regime_mask) - 1):
                if regime_mask[i]:  # If current period is this regime
                    following_returns.append(returns.iloc[i + 1])  # Next period return
            
            if following_returns:
                following_returns = np.array(following_returns)
                
                # Calculate influence metrics
                mean_following_return = np.mean(following_returns)
                volatility_following = np.std(following_returns)
                
                # Information ratio of following returns
                info_ratio = abs(mean_following_return) / volatility_following if volatility_following > 0 else 0
                
                regime_price_influence[regime] = {
                    'mean_following_return': float(mean_following_return),
                    'volatility_following': float(volatility_following),
                    'information_ratio': float(info_ratio),
                    'n_observations': len(following_returns)
                }
            else:
                regime_price_influence[regime] = {
                    'mean_following_return': 0.0,
                    'volatility_following': 0.0,
                    'information_ratio': 0.0,
                    'n_observations': 0
                }
        
        # Calculate overall price action influence
        if len(regime_price_influence) > 1:
            info_ratios = [data['information_ratio'] for data in regime_price_influence.values()]
            max_influence = max(info_ratios) if info_ratios else 0
            
            # Economic significance: >0.1 information ratio difference
            economically_significant = max_influence > 0.1
            
            interpretation = f"Maximum price action influence: {max_influence:.3f} information ratio"
            trading_implications = "Regimes show predictive power for price action" if economically_significant else "Limited price action predictability from regimes"
        else:
            max_influence = 0.0
            economically_significant = False
            interpretation = "Single regime - no price action comparison"
            trading_implications = "No price action benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.PRICE_ACTION_INFLUENCE,
            value=float(max_influence),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['information_ratio'] for k, v in regime_price_influence.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_price_influence}
        )
    
    def _calculate_momentum_prediction_power(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate regime power to predict momentum continuation vs reversal."""
        
        # Calculate momentum indicators
        returns = market_data['close'].pct_change().fillna(0)
        
        # Short-term momentum (5-period)
        momentum_5 = returns.rolling(5).mean()
        
        # Medium-term momentum (20-period) 
        momentum_20 = returns.rolling(20).mean()
        
        unique_regimes = np.unique(regime_labels)
        regime_momentum_power = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            # Get momentum continuation/reversal patterns
            momentum_continuations = []
            momentum_reversals = []
            
            for i in range(20, len(regime_mask) - 5):  # Need lookback and lookahead
                if regime_mask[i]:  # Current regime
                    current_momentum = momentum_5.iloc[i]
                    future_momentum = momentum_5.iloc[i + 5]  # 5 periods ahead
                    
                    if abs(current_momentum) > 0.001:  # Only consider significant momentum
                        # Check if momentum continues or reverses
                        if (current_momentum > 0 and future_momentum > 0) or (current_momentum < 0 and future_momentum < 0):
                            momentum_continuations.append(abs(future_momentum))
                        else:
                            momentum_reversals.append(abs(future_momentum))
            
            # Calculate momentum prediction metrics
            if momentum_continuations or momentum_reversals:
                continuation_strength = np.mean(momentum_continuations) if momentum_continuations else 0
                reversal_strength = np.mean(momentum_reversals) if momentum_reversals else 0
                
                # Momentum prediction power = ability to predict continuation vs reversal
                total_predictions = len(momentum_continuations) + len(momentum_reversals)
                continuation_rate = len(momentum_continuations) / total_predictions if total_predictions > 0 else 0.5
                
                # Strength of prediction (how different from random 50/50)
                prediction_strength = abs(continuation_rate - 0.5) * 2  # Scale to 0-1
                
                regime_momentum_power[regime] = {
                    'continuation_rate': float(continuation_rate),
                    'continuation_strength': float(continuation_strength),
                    'reversal_strength': float(reversal_strength),
                    'prediction_strength': float(prediction_strength),
                    'n_predictions': total_predictions
                }
            else:
                regime_momentum_power[regime] = {
                    'continuation_rate': 0.5,
                    'continuation_strength': 0.0,
                    'reversal_strength': 0.0,
                    'prediction_strength': 0.0,
                    'n_predictions': 0
                }
        
        # Calculate overall momentum prediction power
        if len(regime_momentum_power) > 1:
            prediction_strengths = [data['prediction_strength'] for data in regime_momentum_power.values()]
            max_prediction_power = max(prediction_strengths) if prediction_strengths else 0
            
            # Economic significance: >0.2 prediction strength (significantly better than random)
            economically_significant = max_prediction_power > 0.2
            
            interpretation = f"Maximum momentum prediction power: {max_prediction_power:.3f}"
            trading_implications = "Regimes provide momentum trading signals" if economically_significant else "Limited momentum prediction from regimes"
        else:
            max_prediction_power = 0.0
            economically_significant = False
            interpretation = "Single regime - no momentum prediction comparison"
            trading_implications = "No momentum trading benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.MOMENTUM_PREDICTION_POWER,
            value=float(max_prediction_power),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['prediction_strength'] for k, v in regime_momentum_power.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_momentum_power}
        )
    
    def _calculate_mean_reversion_signal_strength(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate regime strength for mean reversion signals."""
        
        # Calculate mean reversion indicators
        returns = market_data['close'].pct_change().fillna(0)
        prices = market_data['close']
        
        # Calculate deviation from moving average (mean reversion signal)
        ma_20 = prices.rolling(20).mean()
        price_deviation = (prices - ma_20) / ma_20
        
        unique_regimes = np.unique(regime_labels)
        regime_reversion_strength = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            # Analyze mean reversion patterns in this regime
            reversion_signals = []
            reversion_outcomes = []
            
            for i in range(20, len(regime_mask) - 10):  # Need lookback and lookahead
                if regime_mask[i]:  # Current regime
                    current_deviation = price_deviation.iloc[i]
                    
                    if abs(current_deviation) > 0.02:  # Only consider significant deviations (>2%)
                        # Look at reversion over next 10 periods
                        future_prices = prices.iloc[i+1:i+11]
                        current_price = prices.iloc[i]
                        target_price = ma_20.iloc[i]  # Mean to revert to
                        
                        # Calculate reversion strength
                        if current_deviation > 0:  # Price above mean
                            # Look for reversion down
                            min_future_price = future_prices.min()
                            reversion_amount = (current_price - min_future_price) / current_price
                        else:  # Price below mean
                            # Look for reversion up
                            max_future_price = future_prices.max()
                            reversion_amount = (max_future_price - current_price) / current_price
                        
                        reversion_signals.append(abs(current_deviation))
                        reversion_outcomes.append(reversion_amount)
            
            # Calculate mean reversion strength
            if reversion_signals and reversion_outcomes:
                # Correlation between signal strength and reversion outcome
                if len(reversion_signals) > 5:
                    correlation = np.corrcoef(reversion_signals, reversion_outcomes)[0, 1]
                    correlation = correlation if not np.isnan(correlation) else 0
                else:
                    correlation = 0
                
                mean_reversion_strength = abs(correlation)
                avg_reversion_outcome = np.mean(reversion_outcomes)
                
                regime_reversion_strength[regime] = {
                    'signal_reversion_correlation': float(correlation),
                    'mean_reversion_strength': float(mean_reversion_strength),
                    'avg_reversion_outcome': float(avg_reversion_outcome),
                    'n_signals': len(reversion_signals)
                }
            else:
                regime_reversion_strength[regime] = {
                    'signal_reversion_correlation': 0.0,
                    'mean_reversion_strength': 0.0,
                    'avg_reversion_outcome': 0.0,
                    'n_signals': 0
                }
        
        # Calculate overall mean reversion signal strength
        if len(regime_reversion_strength) > 1:
            reversion_strengths = [data['mean_reversion_strength'] for data in regime_reversion_strength.values()]
            max_reversion_strength = max(reversion_strengths) if reversion_strengths else 0
            
            # Economic significance: >0.3 correlation between signals and outcomes
            economically_significant = max_reversion_strength > 0.3
            
            interpretation = f"Maximum mean reversion signal strength: {max_reversion_strength:.3f}"
            trading_implications = "Regimes provide mean reversion trading signals" if economically_significant else "Limited mean reversion signals from regimes"
        else:
            max_reversion_strength = 0.0
            economically_significant = False
            interpretation = "Single regime - no mean reversion comparison"
            trading_implications = "No mean reversion benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.MEAN_REVERSION_SIGNAL_STRENGTH,
            value=float(max_reversion_strength),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['mean_reversion_strength'] for k, v in regime_reversion_strength.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_reversion_strength}
        )
    
    def generate_economic_report(self, 
                               validation_results: Dict[EconomicMetric, EconomicValidationResult]) -> str:
        """Generate comprehensive economic validation report."""
        report = []
        report.append("# Economic Validation Report for Market Regimes")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        economically_significant_metrics = [
            result for result in validation_results.values() 
            if result.economic_significance
        ]
        
        report.append("## Executive Summary")
        report.append("")
        report.append(f"- **Total Metrics Evaluated**: {len(validation_results)}")
        report.append(f"- **Economically Significant**: {len(economically_significant_metrics)}")
        report.append(f"- **Economic Significance Rate**: {len(economically_significant_metrics)/len(validation_results)*100:.1f}%")
        report.append("")
        
        # Key findings
        if economically_significant_metrics:
            report.append("## Key Economic Findings")
            report.append("")
            for result in economically_significant_metrics:
                report.append(f"✅ **{result.metric.value.replace('_', ' ').title()}**")
                report.append(f"   - Value: {result.value:.3f}")
                report.append(f"   - {result.interpretation}")
                report.append(f"   - Trading Implication: {result.trading_implications}")
                report.append("")
        
        # Detailed results by category
        categories = {
            "Risk-Return Analysis": [
                EconomicMetric.SHARPE_RATIO_DIFFERENCE,
                EconomicMetric.INFORMATION_RATIO_DIFFERENCE,
                EconomicMetric.RETURN_SEPARABILITY,
                EconomicMetric.RISK_ADJUSTED_RETURN_DIFF
            ],
            "Risk Management": [
                EconomicMetric.MAXIMUM_DRAWDOWN_DIFFERENCE,
                EconomicMetric.VALUE_AT_RISK_DIFFERENCE
            ],
            "Market Microstructure": [
                EconomicMetric.VOLUME_PROFILE_DIFFERENCE,
                EconomicMetric.LIQUIDITY_COST_DIFFERENCE
            ],
            "Trading Economics": [
                EconomicMetric.REGIME_PERSISTENCE_VALUE,
                EconomicMetric.TRANSITION_COST_ANALYSIS
            ]
        }
        
        for category, metrics in categories.items():
            category_results = [validation_results[m] for m in metrics if m in validation_results]
            if category_results:
                report.append(f"## {category}")
                report.append("")
                
                for result in category_results:
                    status = "✅" if result.economic_significance else "⚠️"
                    report.append(f"{status} **{result.metric.value.replace('_', ' ').title()}**")
                    report.append(f"   - **Value**: {result.value:.3f}")
                    report.append(f"   - **Economic Significance**: {'Yes' if result.economic_significance else 'No'}")
                    if result.statistical_significance:
                        report.append(f"   - **Statistical Significance**: p={result.statistical_significance:.3f}")
                    report.append(f"   - **Interpretation**: {result.interpretation}")
                    report.append(f"   - **Trading Implications**: {result.trading_implications}")
                    
                    # Regime-specific values
                    if result.regime_specific_values:
                        report.append("   - **Regime-Specific Values**:")
                        for regime, value in result.regime_specific_values.items():
                            report.append(f"     - Regime {regime}: {value:.3f}")
                    
                    report.append("")
        
        # Recommendations
        report.append("## Recommendations for ML Model Training")
        report.append("")
        
        significant_count = len(economically_significant_metrics)
        total_count = len(validation_results)
        
        if significant_count >= total_count * 0.7:
            report.append("✅ **Strong Economic Foundation for Regime-Based ML Models**")
            report.append("- Regimes show significant economic differences")
            report.append("- Proceed with regime-specific ML model training")
            report.append("- Focus on economically significant dimensions")
        elif significant_count >= total_count * 0.4:
            report.append("⚠️ **Moderate Economic Foundation**")
            report.append("- Some economic significance detected")
            report.append("- Consider selective regime-based modeling")
            report.append("- Focus on most significant metrics")
        else:
            report.append("❌ **Weak Economic Foundation**")
            report.append("- Limited economic differences between regimes")
            report.append("- Consider regime redefinition or single-model approach")
            report.append("- Investigate alternative clustering methods")
        
        return "\n".join(report)