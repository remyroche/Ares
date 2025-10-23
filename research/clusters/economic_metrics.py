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
    
    # General Price Action Influence Metrics
    PRICE_INSTABILITY_INFLUENCE = "price_instability_influence"
    TREND_DURATION_IMPACT = "trend_duration_impact"
    REVERSAL_VIOLENCE_MODULATION = "reversal_violence_modulation"
    MOMENTUM_INTENSITY_EFFECT = "momentum_intensity_effect"
    TREND_ACCELERATION_IMPACT = "trend_acceleration_impact"
    PRICE_REGIME_TRANSITION_TRIGGER = "price_regime_transition_trigger"
    
    # Missing Critical Metrics
    ASYMMETRIC_VOLATILITY_RESPONSE = "asymmetric_volatility_response"
    REGIME_PERSISTENCE_SCORE = "regime_persistence_score"
    TAIL_DEPENDENCE_INTENSITY = "tail_dependence_intensity"
    
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
        """Calculate Sharpe ratio differences between regimes using VectorBT."""
        regime_sharpe_ratios = {}
        
        try:
            import vectorbt as vbt
            from vectorbt.returns import Returns
            
            for regime, returns in regime_returns.items():
                if len(returns) > 0:
                    # Use VectorBT for Sharpe ratio calculation
                    returns_obj = Returns(returns)
                    sharpe_ratio = returns_obj.sharpe_ratio()
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
                
        except Exception as e:
            logger.warning(f"VectorBT Sharpe ratio calculation failed, using manual calculation: {e}")
            # Fallback to manual calculation
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
        
        # General price action influence metrics
        if 'close' in market_data.columns:
            results[EconomicMetric.PRICE_INSTABILITY_INFLUENCE] = self._calculate_price_instability_influence(market_data, regime_labels)
            results[EconomicMetric.TREND_DURATION_IMPACT] = self._calculate_trend_duration_impact(market_data, regime_labels)
            results[EconomicMetric.REVERSAL_VIOLENCE_MODULATION] = self._calculate_reversal_violence_modulation(market_data, regime_labels)
            results[EconomicMetric.MOMENTUM_INTENSITY_EFFECT] = self._calculate_momentum_intensity_effect(market_data, regime_labels)
            results[EconomicMetric.TREND_ACCELERATION_IMPACT] = self._calculate_trend_acceleration_impact(market_data, regime_labels)
            results[EconomicMetric.PRICE_REGIME_TRANSITION_TRIGGER] = self._calculate_price_regime_transition_trigger(market_data, regime_labels)
            
            # Missing critical metrics
            results[EconomicMetric.ASYMMETRIC_VOLATILITY_RESPONSE] = self._calculate_asymmetric_volatility_response(market_data, regime_labels)
            results[EconomicMetric.REGIME_PERSISTENCE_SCORE] = self._calculate_regime_persistence_score(market_data, regime_labels)
            results[EconomicMetric.TAIL_DEPENDENCE_INTENSITY] = self._calculate_tail_dependence_intensity(market_data, regime_labels)
        
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
    
    def _calculate_price_instability_influence(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate how regimes influence price instability (volatility of volatility, extreme moves)."""
        
        returns = market_data['close'].pct_change().fillna(0)
        
        # Calculate various instability measures
        volatility_20 = returns.rolling(20).std()
        volatility_of_volatility = volatility_20.rolling(20).std()  # Vol of vol
        extreme_moves = (abs(returns) > returns.rolling(100).quantile(0.95)).astype(int)  # Top 5% moves
        
        unique_regimes = np.unique(regime_labels)
        regime_instability = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            if np.sum(regime_mask) > 20:  # Need sufficient data
                regime_vol_of_vol = volatility_of_volatility[regime_mask].mean()
                regime_extreme_freq = extreme_moves[regime_mask].mean()
                regime_return_kurtosis = returns[regime_mask].kurtosis()  # Tail thickness
                
                # Composite instability score
                instability_score = (
                    regime_vol_of_vol * 10 +  # Scale vol of vol
                    regime_extreme_freq * 5 +  # Scale extreme move frequency
                    max(0, regime_return_kurtosis - 3) * 0.1  # Excess kurtosis
                )
                
                regime_instability[regime] = {
                    'volatility_of_volatility': float(regime_vol_of_vol),
                    'extreme_move_frequency': float(regime_extreme_freq),
                    'return_kurtosis': float(regime_return_kurtosis),
                    'instability_score': float(instability_score)
                }
            else:
                regime_instability[regime] = {
                    'volatility_of_volatility': 0.0,
                    'extreme_move_frequency': 0.0,
                    'return_kurtosis': 0.0,
                    'instability_score': 0.0
                }
        
        # Calculate overall instability influence
        if len(regime_instability) > 1:
            instability_scores = [data['instability_score'] for data in regime_instability.values()]
            max_diff = max(instability_scores) - min(instability_scores)
            
            # Economic significance: >0.1 instability difference
            economically_significant = max_diff > 0.1
            
            interpretation = f"Price instability influence: {max_diff:.3f} difference"
            trading_implications = "Regimes significantly affect market instability and extreme moves" if economically_significant else "Similar instability patterns across regimes"
        else:
            max_diff = 0.0
            economically_significant = False
            interpretation = "Single regime - no instability comparison"
            trading_implications = "No instability benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.PRICE_INSTABILITY_INFLUENCE,
            value=float(max_diff),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['instability_score'] for k, v in regime_instability.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_instability}
        )
    
    def _calculate_trend_duration_impact(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate how regimes impact trend duration (how long trends last)."""
        
        prices = market_data['close']
        
        # Define trend using multiple timeframes
        ma_short = prices.rolling(10).mean()
        ma_long = prices.rolling(50).mean()
        trend_direction = np.where(ma_short > ma_long, 1, -1)  # 1=uptrend, -1=downtrend
        
        unique_regimes = np.unique(regime_labels)
        regime_trend_durations = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            # Calculate trend durations within this regime
            trend_durations = []
            current_trend = None
            current_duration = 0
            
            for i in range(len(regime_mask)):
                if regime_mask[i]:  # In this regime
                    if trend_direction[i] == current_trend:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            trend_durations.append(current_duration)
                        current_trend = trend_direction[i]
                        current_duration = 1
                else:
                    # Regime ended, record current trend if any
                    if current_duration > 0:
                        trend_durations.append(current_duration)
                        current_duration = 0
                        current_trend = None
            
            # Final trend
            if current_duration > 0:
                trend_durations.append(current_duration)
            
            # Calculate trend duration statistics
            if trend_durations:
                avg_duration = np.mean(trend_durations)
                median_duration = np.median(trend_durations)
                max_duration = np.max(trend_durations)
                duration_volatility = np.std(trend_durations)
                
                regime_trend_durations[regime] = {
                    'avg_trend_duration': float(avg_duration),
                    'median_trend_duration': float(median_duration),
                    'max_trend_duration': float(max_duration),
                    'duration_volatility': float(duration_volatility),
                    'n_trends': len(trend_durations)
                }
            else:
                regime_trend_durations[regime] = {
                    'avg_trend_duration': 0.0,
                    'median_trend_duration': 0.0,
                    'max_trend_duration': 0.0,
                    'duration_volatility': 0.0,
                    'n_trends': 0
                }
        
        # Calculate trend duration impact
        if len(regime_trend_durations) > 1:
            avg_durations = [data['avg_trend_duration'] for data in regime_trend_durations.values()]
            duration_range = max(avg_durations) - min(avg_durations)
            
            # Economic significance: >5 period difference in trend duration
            economically_significant = duration_range > 5.0
            
            interpretation = f"Trend duration impact: {duration_range:.1f} period difference"
            trading_implications = "Regimes significantly affect trend persistence and duration" if economically_significant else "Similar trend durations across regimes"
        else:
            duration_range = 0.0
            economically_significant = False
            interpretation = "Single regime - no trend duration comparison"
            trading_implications = "No trend duration benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.TREND_DURATION_IMPACT,
            value=float(duration_range),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['avg_trend_duration'] for k, v in regime_trend_durations.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_trend_durations}
        )
    
    def _calculate_reversal_violence_modulation(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate how regimes modulate reversal violence (speed and magnitude of reversals)."""
        
        prices = market_data['close']
        returns = prices.pct_change().fillna(0)
        
        # Detect reversals using local extrema
        ma_10 = prices.rolling(10).mean()
        price_position = (prices - ma_10) / ma_10  # Position relative to MA
        
        unique_regimes = np.unique(regime_labels)
        regime_reversal_violence = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            # Find reversals within this regime
            reversal_magnitudes = []
            reversal_speeds = []
            
            for i in range(10, len(regime_mask) - 10):
                if regime_mask[i]:  # In this regime
                    current_position = price_position.iloc[i]
                    
                    # Look for significant position (>2% from MA)
                    if abs(current_position) > 0.02:
                        # Look for reversal in next 10 periods
                        future_positions = price_position.iloc[i+1:i+11]
                        
                        # Find if reversal occurs (crossing zero)
                        if current_position > 0:  # Above MA, look for move below
                            reversal_points = future_positions[future_positions < 0]
                        else:  # Below MA, look for move above
                            reversal_points = future_positions[future_positions > 0]
                        
                        if len(reversal_points) > 0:
                            # Calculate reversal magnitude and speed
                            reversal_magnitude = abs(current_position - reversal_points.iloc[0])
                            reversal_speed = reversal_magnitude / (np.where(future_positions.index == reversal_points.index[0])[0][0] + 1)
                            
                            reversal_magnitudes.append(reversal_magnitude)
                            reversal_speeds.append(reversal_speed)
            
            # Calculate reversal violence metrics
            if reversal_magnitudes and reversal_speeds:
                avg_magnitude = np.mean(reversal_magnitudes)
                avg_speed = np.mean(reversal_speeds)
                violence_score = avg_magnitude * avg_speed  # Magnitude × Speed = Violence
                
                regime_reversal_violence[regime] = {
                    'avg_reversal_magnitude': float(avg_magnitude),
                    'avg_reversal_speed': float(avg_speed),
                    'reversal_violence_score': float(violence_score),
                    'n_reversals': len(reversal_magnitudes)
                }
            else:
                regime_reversal_violence[regime] = {
                    'avg_reversal_magnitude': 0.0,
                    'avg_reversal_speed': 0.0,
                    'reversal_violence_score': 0.0,
                    'n_reversals': 0
                }
        
        # Calculate reversal violence modulation
        if len(regime_reversal_violence) > 1:
            violence_scores = [data['reversal_violence_score'] for data in regime_reversal_violence.values()]
            violence_range = max(violence_scores) - min(violence_scores)
            
            # Economic significance: >0.001 violence difference (magnitude × speed)
            economically_significant = violence_range > 0.001
            
            interpretation = f"Reversal violence modulation: {violence_range:.4f} difference"
            trading_implications = "Regimes significantly affect reversal speed and magnitude" if economically_significant else "Similar reversal patterns across regimes"
        else:
            violence_range = 0.0
            economically_significant = False
            interpretation = "Single regime - no reversal violence comparison"
            trading_implications = "No reversal pattern benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.REVERSAL_VIOLENCE_MODULATION,
            value=float(violence_range),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['reversal_violence_score'] for k, v in regime_reversal_violence.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_reversal_violence}
        )
    
    def _calculate_momentum_intensity_effect(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate how regimes affect momentum intensity (strength of price movements)."""
        
        returns = market_data['close'].pct_change().fillna(0)
        
        # Calculate momentum intensity measures
        momentum_5 = returns.rolling(5).mean()
        momentum_20 = returns.rolling(20).mean()
        momentum_strength = abs(momentum_5) + abs(momentum_20)  # Combined momentum strength
        
        # Calculate acceleration (rate of change of momentum)
        momentum_acceleration = momentum_5.diff()
        
        unique_regimes = np.unique(regime_labels)
        regime_momentum_intensity = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            if np.sum(regime_mask) > 20:
                regime_momentum_data = momentum_strength[regime_mask]
                regime_acceleration_data = momentum_acceleration[regime_mask]
                
                # Calculate intensity metrics
                avg_intensity = regime_momentum_data.mean()
                max_intensity = regime_momentum_data.quantile(0.95)  # 95th percentile
                intensity_volatility = regime_momentum_data.std()
                
                # Acceleration metrics
                avg_acceleration = abs(regime_acceleration_data).mean()
                max_acceleration = abs(regime_acceleration_data).quantile(0.95)
                
                # Composite intensity effect
                intensity_effect = avg_intensity + max_intensity * 0.5 + avg_acceleration * 10
                
                regime_momentum_intensity[regime] = {
                    'avg_momentum_intensity': float(avg_intensity),
                    'max_momentum_intensity': float(max_intensity),
                    'intensity_volatility': float(intensity_volatility),
                    'avg_acceleration': float(avg_acceleration),
                    'max_acceleration': float(max_acceleration),
                    'composite_intensity_effect': float(intensity_effect)
                }
            else:
                regime_momentum_intensity[regime] = {
                    'avg_momentum_intensity': 0.0,
                    'max_momentum_intensity': 0.0,
                    'intensity_volatility': 0.0,
                    'avg_acceleration': 0.0,
                    'max_acceleration': 0.0,
                    'composite_intensity_effect': 0.0
                }
        
        # Calculate momentum intensity effect
        if len(regime_momentum_intensity) > 1:
            intensity_effects = [data['composite_intensity_effect'] for data in regime_momentum_intensity.values()]
            intensity_range = max(intensity_effects) - min(intensity_effects)
            
            # Economic significance: >0.01 intensity difference
            economically_significant = intensity_range > 0.01
            
            interpretation = f"Momentum intensity effect: {intensity_range:.4f} difference"
            trading_implications = "Regimes significantly affect momentum strength and acceleration" if economically_significant else "Similar momentum patterns across regimes"
        else:
            intensity_range = 0.0
            economically_significant = False
            interpretation = "Single regime - no momentum intensity comparison"
            trading_implications = "No momentum intensity benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.MOMENTUM_INTENSITY_EFFECT,
            value=float(intensity_range),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['composite_intensity_effect'] for k, v in regime_momentum_intensity.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_momentum_intensity}
        )
    
    def _calculate_trend_acceleration_impact(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate how regimes impact trend acceleration (rate of change of trend strength)."""
        
        prices = market_data['close']
        returns = prices.pct_change().fillna(0)
        
        # Calculate trend strength using linear regression slope
        trend_strengths = []
        for i in range(20, len(prices)):
            window_prices = prices.iloc[i-20:i]
            if len(window_prices) == 20:
                # Linear regression slope as trend strength
                x = np.arange(20)
                slope = np.polyfit(x, window_prices, 1)[0]
                trend_strength = slope / window_prices.iloc[-1]  # Normalize by price
                trend_strengths.append(trend_strength)
            else:
                trend_strengths.append(0.0)
        
        # Pad to match original length
        trend_strengths = [0.0] * 20 + trend_strengths
        trend_strength_series = pd.Series(trend_strengths, index=prices.index)
        
        # Calculate trend acceleration (change in trend strength)
        trend_acceleration = trend_strength_series.diff()
        
        unique_regimes = np.unique(regime_labels)
        regime_acceleration_impact = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            if np.sum(regime_mask) > 20:
                regime_acceleration_data = trend_acceleration[regime_mask]
                
                # Calculate acceleration metrics
                avg_acceleration = abs(regime_acceleration_data).mean()
                max_acceleration = abs(regime_acceleration_data).quantile(0.95)
                acceleration_volatility = regime_acceleration_data.std()
                
                # Trend strength metrics
                regime_strength_data = trend_strength_series[regime_mask]
                avg_trend_strength = abs(regime_strength_data).mean()
                
                # Composite acceleration impact
                acceleration_impact = avg_acceleration + max_acceleration * 0.5 + avg_trend_strength
                
                regime_acceleration_impact[regime] = {
                    'avg_acceleration': float(avg_acceleration),
                    'max_acceleration': float(max_acceleration),
                    'acceleration_volatility': float(acceleration_volatility),
                    'avg_trend_strength': float(avg_trend_strength),
                    'acceleration_impact': float(acceleration_impact)
                }
            else:
                regime_acceleration_impact[regime] = {
                    'avg_acceleration': 0.0,
                    'max_acceleration': 0.0,
                    'acceleration_volatility': 0.0,
                    'avg_trend_strength': 0.0,
                    'acceleration_impact': 0.0
                }
        
        # Calculate trend acceleration impact
        if len(regime_acceleration_impact) > 1:
            acceleration_impacts = [data['acceleration_impact'] for data in regime_acceleration_impact.values()]
            impact_range = max(acceleration_impacts) - min(acceleration_impacts)
            
            # Economic significance: >0.001 acceleration difference
            economically_significant = impact_range > 0.001
            
            interpretation = f"Trend acceleration impact: {impact_range:.4f} difference"
            trading_implications = "Regimes significantly affect trend acceleration and strength" if economically_significant else "Similar trend acceleration across regimes"
        else:
            impact_range = 0.0
            economically_significant = False
            interpretation = "Single regime - no acceleration comparison"
            trading_implications = "No trend acceleration benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.TREND_ACCELERATION_IMPACT,
            value=float(impact_range),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['acceleration_impact'] for k, v in regime_acceleration_impact.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_acceleration_impact}
        )
    
    def _calculate_price_regime_transition_trigger(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """Calculate how price action triggers regime transitions."""
        
        returns = market_data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std()
        
        # Detect regime transitions
        regime_changes = np.diff(regime_labels) != 0
        regime_change_indices = np.where(regime_changes)[0] + 1  # +1 to get the new regime start
        
        # Analyze price conditions around regime transitions
        transition_triggers = []
        
        for change_idx in regime_change_indices:
            if change_idx >= 10 and change_idx < len(returns) - 5:
                # Price conditions before transition (5 periods before)
                pre_returns = returns.iloc[change_idx-5:change_idx]
                pre_volatility = volatility.iloc[change_idx-5:change_idx]
                
                # Price conditions after transition (5 periods after)
                post_returns = returns.iloc[change_idx:change_idx+5]
                post_volatility = volatility.iloc[change_idx:change_idx+5]
                
                # Calculate trigger characteristics
                pre_extreme_move = any(abs(pre_returns) > pre_returns.rolling(20).quantile(0.95).iloc[-1])
                pre_vol_spike = any(pre_volatility > pre_volatility.rolling(20).quantile(0.95).iloc[-1])
                
                post_behavior_change = abs(post_returns.mean() - pre_returns.mean())
                post_vol_change = abs(post_volatility.mean() - pre_volatility.mean())
                
                transition_triggers.append({
                    'pre_extreme_move': pre_extreme_move,
                    'pre_vol_spike': pre_vol_spike,
                    'post_behavior_change': post_behavior_change,
                    'post_vol_change': post_vol_change,
                    'trigger_strength': float(pre_extreme_move) + float(pre_vol_spike) + post_behavior_change * 100 + post_vol_change * 100
                })
        
        # Calculate transition trigger characteristics
        if transition_triggers:
            avg_trigger_strength = np.mean([t['trigger_strength'] for t in transition_triggers])
            extreme_move_rate = np.mean([t['pre_extreme_move'] for t in transition_triggers])
            vol_spike_rate = np.mean([t['pre_vol_spike'] for t in transition_triggers])
            avg_behavior_change = np.mean([t['post_behavior_change'] for t in transition_triggers])
            
            # Economic significance: >0.5 trigger strength or >50% extreme move rate
            economically_significant = avg_trigger_strength > 0.5 or extreme_move_rate > 0.5
            
            interpretation = f"Regime transitions triggered by extreme moves {extreme_move_rate:.1%} of time"
            trading_implications = "Price action significantly triggers regime changes" if economically_significant else "Regime changes not strongly price-driven"
        else:
            avg_trigger_strength = 0.0
            economically_significant = False
            interpretation = "No regime transitions detected"
            trading_implications = "Stable regime - no transition analysis possible"
        
        return EconomicValidationResult(
            metric=EconomicMetric.PRICE_REGIME_TRANSITION_TRIGGER,
            value=float(avg_trigger_strength),
            economic_significance=economically_significant,
            regime_specific_values={'avg_trigger_strength': avg_trigger_strength},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'transition_details': transition_triggers[:10] if transition_triggers else []}  # Limit for storage
        )
    
    def _calculate_asymmetric_volatility_response(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """
        Calculate asymmetric volatility response (leverage effect).
        
        Measures whether downside moves produce stronger volatility responses than upside moves.
        Critical for tail-risk hedging and options pricing.
        """
        
        returns = market_data['close'].pct_change().fillna(0)
        
        # Calculate realized volatility
        volatility = returns.rolling(20).std()
        
        unique_regimes = np.unique(regime_labels)
        regime_asymmetric_responses = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            if np.sum(regime_mask) > 50:  # Need sufficient data for asymmetry analysis
                regime_returns = returns[regime_mask]
                regime_volatility = volatility[regime_mask]
                
                # Separate positive and negative returns
                positive_returns = regime_returns[regime_returns > 0]
                negative_returns = regime_returns[regime_returns < 0]
                
                # Calculate volatility following positive vs negative returns
                vol_after_positive = []
                vol_after_negative = []
                
                for i in range(len(regime_returns) - 1):
                    current_return = regime_returns.iloc[i]
                    next_volatility = regime_volatility.iloc[i + 1] if i + 1 < len(regime_volatility) else np.nan
                    
                    if not np.isnan(next_volatility):
                        if current_return > 0:
                            vol_after_positive.append(next_volatility)
                        elif current_return < 0:
                            vol_after_negative.append(next_volatility)
                
                # Calculate asymmetric response
                if vol_after_positive and vol_after_negative:
                    avg_vol_after_positive = np.mean(vol_after_positive)
                    avg_vol_after_negative = np.mean(vol_after_negative)
                    
                    # Asymmetry ratio: vol_after_negative / vol_after_positive
                    # >1 = leverage effect (downside increases vol more)
                    asymmetry_ratio = avg_vol_after_negative / avg_vol_after_positive if avg_vol_after_positive > 0 else 1.0
                    
                    # Return skewness (additional asymmetry measure)
                    return_skewness = regime_returns.skew()
                    
                    # Composite asymmetric response score
                    asymmetric_response = abs(asymmetry_ratio - 1.0) + abs(return_skewness) * 0.1
                    
                    regime_asymmetric_responses[regime] = {
                        'asymmetry_ratio': float(asymmetry_ratio),
                        'return_skewness': float(return_skewness),
                        'vol_after_positive': float(avg_vol_after_positive),
                        'vol_after_negative': float(avg_vol_after_negative),
                        'asymmetric_response_score': float(asymmetric_response)
                    }
                else:
                    regime_asymmetric_responses[regime] = {
                        'asymmetry_ratio': 1.0,
                        'return_skewness': 0.0,
                        'vol_after_positive': 0.0,
                        'vol_after_negative': 0.0,
                        'asymmetric_response_score': 0.0
                    }
            else:
                regime_asymmetric_responses[regime] = {
                    'asymmetry_ratio': 1.0,
                    'return_skewness': 0.0,
                    'vol_after_positive': 0.0,
                    'vol_after_negative': 0.0,
                    'asymmetric_response_score': 0.0
                }
        
        # Calculate overall asymmetric volatility response difference
        if len(regime_asymmetric_responses) > 1:
            response_scores = [data['asymmetric_response_score'] for data in regime_asymmetric_responses.values()]
            max_asymmetry_diff = max(response_scores) - min(response_scores)
            
            # Economic significance: >0.2 asymmetry difference (20% leverage effect difference)
            economically_significant = max_asymmetry_diff > 0.2
            
            interpretation = f"Asymmetric volatility response difference: {max_asymmetry_diff:.3f}"
            trading_implications = "Regimes show different leverage effects - important for tail hedging and options" if economically_significant else "Similar volatility asymmetry across regimes"
        else:
            max_asymmetry_diff = 0.0
            economically_significant = False
            interpretation = "Single regime - no asymmetry comparison"
            trading_implications = "No asymmetric volatility benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.ASYMMETRIC_VOLATILITY_RESPONSE,
            value=float(max_asymmetry_diff),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['asymmetric_response_score'] for k, v in regime_asymmetric_responses.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_asymmetric_responses}
        )
    
    def _calculate_regime_persistence_score(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """
        Calculate regime persistence score (how sticky regimes are).
        
        Measures how long regimes typically last and their transition probabilities.
        Critical for strategy confidence and commitment levels.
        """
        
        # Calculate regime durations and transition matrix
        regime_durations = []
        current_regime = regime_labels[0]
        current_duration = 1
        
        # Track regime durations
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append((current_regime, current_duration))
                current_regime = regime_labels[i]
                current_duration = 1
        
        # Add final regime
        regime_durations.append((current_regime, current_duration))
        
        # Calculate transition matrix
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_labels) - 1):
            from_regime_idx = np.where(unique_regimes == regime_labels[i])[0][0]
            to_regime_idx = np.where(unique_regimes == regime_labels[i + 1])[0][0]
            transition_matrix[from_regime_idx, to_regime_idx] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, 
                                   out=np.zeros_like(transition_matrix), 
                                   where=row_sums!=0)
        
        # Calculate persistence metrics for each regime
        regime_persistence_metrics = {}
        
        for i, regime in enumerate(unique_regimes):
            # Duration statistics
            regime_duration_list = [duration for reg, duration in regime_durations if reg == regime]
            
            if regime_duration_list:
                avg_duration = np.mean(regime_duration_list)
                median_duration = np.median(regime_duration_list)
                max_duration = np.max(regime_duration_list)
                
                # Persistence probability (diagonal of transition matrix)
                persistence_prob = transition_probs[i, i] if i < len(transition_probs) else 0.0
                
                # Half-life calculation (how long until 50% chance of regime change)
                if persistence_prob > 0 and persistence_prob < 1:
                    half_life = np.log(0.5) / np.log(persistence_prob)
                else:
                    half_life = avg_duration
                
                # Composite persistence score
                persistence_score = (avg_duration * 0.4 + half_life * 0.4 + persistence_prob * 20)
                
                regime_persistence_metrics[regime] = {
                    'avg_duration': float(avg_duration),
                    'median_duration': float(median_duration),
                    'max_duration': float(max_duration),
                    'persistence_probability': float(persistence_prob),
                    'half_life': float(half_life),
                    'persistence_score': float(persistence_score)
                }
            else:
                regime_persistence_metrics[regime] = {
                    'avg_duration': 0.0,
                    'median_duration': 0.0,
                    'max_duration': 0.0,
                    'persistence_probability': 0.0,
                    'half_life': 0.0,
                    'persistence_score': 0.0
                }
        
        # Calculate overall persistence score difference
        if len(regime_persistence_metrics) > 1:
            persistence_scores = [data['persistence_score'] for data in regime_persistence_metrics.values()]
            persistence_range = max(persistence_scores) - min(persistence_scores)
            
            # Economic significance: >10 persistence score difference
            economically_significant = persistence_range > 10.0
            
            interpretation = f"Regime persistence difference: {persistence_range:.1f} score range"
            trading_implications = "Significant persistence differences enable regime-specific strategy commitment levels" if economically_significant else "Similar persistence across regimes"
        else:
            persistence_range = 0.0
            economically_significant = False
            interpretation = "Single regime - no persistence comparison"
            trading_implications = "No persistence benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.REGIME_PERSISTENCE_SCORE,
            value=float(persistence_range),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['persistence_score'] for k, v in regime_persistence_metrics.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={
                'regime_details': regime_persistence_metrics,
                'transition_matrix': transition_probs.tolist()
            }
        )
    
    def _calculate_tail_dependence_intensity(self, market_data: pd.DataFrame, regime_labels: np.ndarray) -> EconomicValidationResult:
        """
        Calculate tail dependence intensity (clustering of extreme events).
        
        Measures how strongly extreme events cluster within regimes.
        Critical for crisis regime detection and tail hedging.
        """
        
        returns = market_data['close'].pct_change().fillna(0)
        
        # Define extreme events (VaR exceedances)
        var_95 = returns.rolling(100).quantile(0.05)  # 95% VaR (5th percentile)
        var_99 = returns.rolling(100).quantile(0.01)  # 99% VaR (1st percentile)
        
        # Extreme event indicators
        extreme_events_95 = (returns <= var_95).astype(int)
        extreme_events_99 = (returns <= var_99).astype(int)
        
        unique_regimes = np.unique(regime_labels)
        regime_tail_dependence = {}
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            
            if np.sum(regime_mask) > 100:  # Need sufficient data for tail analysis
                regime_returns = returns[regime_mask]
                regime_extreme_95 = extreme_events_95[regime_mask]
                regime_extreme_99 = extreme_events_99[regime_mask]
                
                # Tail dependence measures
                # 1. Clustering coefficient (probability of extreme event following extreme event)
                clustering_95 = self._calculate_clustering_coefficient(regime_extreme_95)
                clustering_99 = self._calculate_clustering_coefficient(regime_extreme_99)
                
                # 2. Tail conditional correlation
                tail_corr = self._calculate_tail_conditional_correlation(regime_returns)
                
                # 3. Extreme event frequency
                extreme_freq_95 = regime_extreme_95.mean()
                extreme_freq_99 = regime_extreme_99.mean()
                
                # 4. Tail thickness (kurtosis of extreme events)
                extreme_returns = regime_returns[regime_returns <= regime_returns.quantile(0.05)]
                tail_kurtosis = extreme_returns.kurtosis() if len(extreme_returns) > 10 else 0
                
                # Composite tail dependence intensity
                tail_intensity = (
                    clustering_95 * 0.3 +
                    clustering_99 * 0.3 +
                    abs(tail_corr) * 0.2 +
                    extreme_freq_95 * 0.1 +
                    max(0, tail_kurtosis - 3) * 0.1  # Excess kurtosis
                )
                
                regime_tail_dependence[regime] = {
                    'clustering_95': float(clustering_95),
                    'clustering_99': float(clustering_99),
                    'tail_conditional_correlation': float(tail_corr),
                    'extreme_frequency_95': float(extreme_freq_95),
                    'extreme_frequency_99': float(extreme_freq_99),
                    'tail_kurtosis': float(tail_kurtosis),
                    'tail_dependence_intensity': float(tail_intensity)
                }
            else:
                regime_tail_dependence[regime] = {
                    'clustering_95': 0.0,
                    'clustering_99': 0.0,
                    'tail_conditional_correlation': 0.0,
                    'extreme_frequency_95': 0.0,
                    'extreme_frequency_99': 0.0,
                    'tail_kurtosis': 0.0,
                    'tail_dependence_intensity': 0.0
                }
        
        # Calculate overall tail dependence intensity difference
        if len(regime_tail_dependence) > 1:
            intensity_scores = [data['tail_dependence_intensity'] for data in regime_tail_dependence.values()]
            intensity_range = max(intensity_scores) - min(intensity_scores)
            
            # Economic significance: >0.1 tail intensity difference
            economically_significant = intensity_range > 0.1
            
            interpretation = f"Tail dependence intensity difference: {intensity_range:.3f}"
            trading_implications = "Regimes show different extreme event clustering - critical for tail hedging" if economically_significant else "Similar tail behavior across regimes"
        else:
            intensity_range = 0.0
            economically_significant = False
            interpretation = "Single regime - no tail dependence comparison"
            trading_implications = "No tail risk benefits from regime identification"
        
        return EconomicValidationResult(
            metric=EconomicMetric.TAIL_DEPENDENCE_INTENSITY,
            value=float(intensity_range),
            economic_significance=economically_significant,
            regime_specific_values={int(k): v['tail_dependence_intensity'] for k, v in regime_tail_dependence.items()},
            statistical_significance=None,
            confidence_interval=None,
            interpretation=interpretation,
            trading_implications=trading_implications,
            metadata={'regime_details': regime_tail_dependence}
        )
    
    def _calculate_clustering_coefficient(self, extreme_events: pd.Series) -> float:
        """Calculate clustering coefficient for extreme events."""
        if len(extreme_events) < 10:
            return 0.0
        
        # Calculate probability of extreme event following extreme event
        clustering_count = 0
        extreme_count = 0
        
        for i in range(len(extreme_events) - 1):
            if extreme_events.iloc[i] == 1:  # Current period is extreme
                extreme_count += 1
                if extreme_events.iloc[i + 1] == 1:  # Next period is also extreme
                    clustering_count += 1
        
        clustering_coefficient = clustering_count / extreme_count if extreme_count > 0 else 0.0
        return float(clustering_coefficient)
    
    def _calculate_tail_conditional_correlation(self, returns: pd.Series) -> float:
        """Calculate tail conditional correlation (correlation in extreme events)."""
        
        # Get extreme negative returns (bottom 5%)
        threshold = returns.quantile(0.05)
        extreme_returns = returns[returns <= threshold]
        
        if len(extreme_returns) < 10:
            return 0.0
        
        # Calculate autocorrelation of extreme returns
        try:
            tail_autocorr = extreme_returns.autocorr(1)
            return float(tail_autocorr) if not np.isnan(tail_autocorr) else 0.0
        except:
            return 0.0
    
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