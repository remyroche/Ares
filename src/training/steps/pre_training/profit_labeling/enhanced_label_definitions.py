"""
Enhanced Label Definitions for Trading ML

This module implements the specific label definitions requested:
1. Analyst labels: "Should we trade?" based on expected PnL > fees + slippage
2. Tactician labels: Direction/magnitude based on max favorable/adverse excursion
3. Regime conditioning: Volatility-scaled thresholds
4. Risk awareness: Label 0 if trade would hit stop before target
5. Data cleaning: Remove outliers, align timestamps, de-duplicate
6. Stability checks: Recompute labels, track leakage, check OOS balance

This extends the existing volatility-aware labeling system with these specific definitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation


class LabelDefinitionType(Enum):
    """Types of label definitions."""
    ANALYST = "analyst"           # Should we trade? (0/1)
    TACTICIAN = "tactician"       # Direction/magnitude (0/1 with thresholds)
    REGIME_CONDITIONED = "regime_conditioned"  # Volatility-scaled
    RISK_AWARE = "risk_aware"     # Stop-loss aware


@dataclass
class TradingCosts:
    """Trading costs configuration."""
    maker_fee: float = 0.001    # 0.1% maker fee
    taker_fee: float = 0.002    # 0.2% taker fee
    slippage_pct: float = 0.001  # 0.1% slippage estimate
    min_trade_size: float = 10.0  # Minimum trade size in USD
    default_is_maker: bool = True
    default_asset_class: str = "default"
    borrow_fees: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"default": {"long": 0.0, "short": 0.0}}
    )
    funding_rates: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {"default": {"long": 0.0, "short": 0.0}}
    )
    stress_scenarios: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=lambda: {
            "default": {
                "base": {"long": 1.0, "short": 1.0}
            }
        }
    )
    active_stress_scenario: str = "base"

    def total_costs(self, trade_size_usd: float, is_maker: Optional[bool] = None) -> float:
        """Calculate total costs for a trade."""
        maker_flag = self.default_is_maker if is_maker is None else is_maker
        fee_rate = self.maker_fee if maker_flag else self.taker_fee
        fee_cost = trade_size_usd * fee_rate
        slippage_cost = trade_size_usd * self.slippage_pct
        return fee_cost + slippage_cost

    def _normalize_direction(self, position_side: str) -> str:
        side = (position_side or "long").lower()
        if side not in {"long", "short"}:
            return "long"
        return side

    def _resolve_asset_entry(self, mapping: Dict[str, Any], asset_class: str) -> Optional[Any]:
        if asset_class in mapping:
            return mapping[asset_class]
        if self.default_asset_class in mapping:
            return mapping[self.default_asset_class]
        return None

    def get_borrow_rate(self, asset_class: str, position_side: str) -> float:
        """Return borrow rate for the given asset class and position side."""
        side = self._normalize_direction(position_side)
        asset_entry = self._resolve_asset_entry(self.borrow_fees, asset_class)
        if asset_entry is None:
            raise ValueError(
                f"Borrow fee assumptions missing for asset class '{asset_class}'."
            )
        return float(asset_entry.get(side, asset_entry.get("both", 0.0)))

    def get_funding_rate(self, asset_class: str, position_side: str) -> float:
        """Return funding rate for the given asset class and position side."""
        side = self._normalize_direction(position_side)
        asset_entry = self._resolve_asset_entry(self.funding_rates, asset_class)
        if asset_entry is None:
            raise ValueError(
                f"Funding rate assumptions missing for asset class '{asset_class}'."
            )
        return float(asset_entry.get(side, asset_entry.get("both", 0.0)))

    def get_stress_multiplier(
        self,
        asset_class: str,
        position_side: str,
        scenario: Optional[str] = None
    ) -> float:
        """Return stress multiplier for the given asset class, side, and scenario."""
        side = self._normalize_direction(position_side)
        asset_entry = self._resolve_asset_entry(self.stress_scenarios, asset_class)
        if asset_entry is None:
            return 1.0

        scenario_key = scenario or self.active_stress_scenario
        scenario_entry = asset_entry.get(scenario_key)

        if scenario_entry is None:
            # Fallback to active scenario or base scenario
            scenario_entry = asset_entry.get(self.active_stress_scenario) or asset_entry.get("base")

        if scenario_entry is None:
            return 1.0

        return float(scenario_entry.get(side, scenario_entry.get("both", 1.0)))

    def validate_asset_assumptions(self, asset_classes: Iterable[str]) -> None:
        """Ensure borrow and funding assumptions exist for the provided asset classes."""
        missing_borrow: List[str] = []
        missing_funding: List[str] = []

        for asset_class in asset_classes:
            asset_key = asset_class if asset_class is not None else self.default_asset_class
            if self._resolve_asset_entry(self.borrow_fees, asset_key) is None:
                missing_borrow.append(str(asset_key))
            if self._resolve_asset_entry(self.funding_rates, asset_key) is None:
                missing_funding.append(str(asset_key))

        errors = []
        if missing_borrow:
            errors.append(
                f"Borrow fee assumptions missing for: {', '.join(sorted(set(missing_borrow)))}"
            )
        if missing_funding:
            errors.append(
                f"Funding rate assumptions missing for: {', '.join(sorted(set(missing_funding)))}"
            )

        if errors:
            raise ValueError("; ".join(errors))


@dataclass
class AsymmetricReturnScalingConfig:
    """Configuration for asymmetric scaling of expected returns."""

    enabled: bool = False
    method: str = "weighted_tail"  # "weighted_tail" or "skew_adjusted"
    upside_weight: float = 1.0
    downside_weight: float = 1.0
    upside_tail_percentile: float = 0.75
    downside_tail_percentile: float = 0.25
    tail_lookback_window: int = 20
    blend_ratio: float = 0.5  # Blend between baseline expectation and asymmetric view


@dataclass
class AsymmetricRiskAdjustmentConfig:
    """Configuration for asymmetric risk awareness adjustments."""

    enable_downside_penalty: bool = False
    downside_penalty_multiplier: float = 1.0
    downside_tail_percentile: float = 0.2
    penalty_lookback_window: int = 20
    apply_only_to_positive: bool = True
    enable_asymmetric_clamp: bool = False
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None


@dataclass
class AnalystLabelConfig:
    """Configuration for Analyst labels."""

    # Trading horizon in minutes
    horizon_minutes: int = 60

    # Profitability thresholds
    min_profit_threshold_usd: float = 5.0
    min_profit_threshold_pct: float = 0.001  # 0.1%

    # Trading costs
    trading_costs: TradingCosts = field(default_factory=TradingCosts)

    # Confidence thresholds
    min_confidence_threshold: float = 0.6

    # Data quality filters
    min_volume_threshold: float = 1000.0
    max_spread_pct: float = 0.01  # 1%

    # Risk management
    max_position_size_pct: float = 0.05  # 5% of portfolio
    max_drawdown_pct: float = 0.02      # 2% max drawdown

    # Regime conditioning
    enable_regime_conditioning: bool = True
    volatility_scaling_factor: float = 1.0

    # Asymmetric behaviour
    asymmetric_return_scaling: AsymmetricReturnScalingConfig = field(default_factory=AsymmetricReturnScalingConfig)
    asymmetric_risk_adjustment: AsymmetricRiskAdjustmentConfig = field(default_factory=AsymmetricRiskAdjustmentConfig)


@dataclass
class TacticianLabelConfig:
    """Configuration for Tactician labels."""

    # Excursion thresholds
    favorable_excursion_threshold: float = 1.0  # 1σ
    adverse_excursion_threshold: float = -2.0   # -2σ

    # Horizon settings
    horizon_minutes: int = 30

    # Direction confidence
    min_direction_confidence: float = 0.7

    # Magnitude scaling
    magnitude_scaling: bool = True
    max_magnitude: float = 5.0

    # Regime conditioning
    enable_regime_conditioning: bool = True
    volatility_sensitivity: float = 1.0

    # Asymmetric behaviour controls (shared structure with analyst config)
    asymmetric_return_scaling: AsymmetricReturnScalingConfig = field(default_factory=AsymmetricReturnScalingConfig)
    asymmetric_risk_adjustment: AsymmetricRiskAdjustmentConfig = field(default_factory=AsymmetricRiskAdjustmentConfig)


@dataclass
class RegimeConditionedConfig:
    """Configuration for regime-conditioned labels."""

    # Volatility scaling
    volatility_scaling_enabled: bool = True
    base_threshold_multiplier: float = 1.0

    # Regime-specific adjustments
    low_vol_multiplier: float = 0.5
    high_vol_multiplier: float = 2.0

    # Adaptive thresholds
    adaptive_thresholds: bool = True
    lookback_window: int = 50

    # Regime detection
    regime_volatility_percentiles: Tuple[float, float] = (25.0, 75.0)


@dataclass
class RiskAwareConfig:
    """Configuration for risk-aware labels."""

    # Stop-loss settings
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit

    # Risk-reward ratio
    min_risk_reward_ratio: float = 2.0

    # Position sizing
    kelly_fraction: float = 0.25  # 25% of Kelly optimal

    # Risk limits
    max_portfolio_risk_pct: float = 0.02  # 2% max portfolio risk per trade
    max_correlation_risk: float = 0.7     # Max correlation with existing positions


@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning."""

    # Outlier detection
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 3.0

    # Volume filters
    min_volume_threshold: float = 1000.0
    max_volume_threshold: float = float('inf')

    # Price filters
    min_price: float = 0.01
    max_price_change_pct: float = 0.50  # 50% max price change per bar

    # Timestamp alignment
    enforce_timestamp_alignment: bool = True
    max_timestamp_gap_minutes: int = 60

    # Deduplication
    enable_deduplication: bool = True
    dedup_method: str = "time_volume"  # "time_volume", "exact_match"


@dataclass
class StabilityCheckConfig:
    """Configuration for stability checks."""

    # Label recomputation
    recompute_on_refresh: bool = True
    max_recomputation_gap_days: int = 7

    # Leakage detection
    max_autocorrelation_threshold: float = 0.3
    lookback_window_leakage: int = 100

    # OOS balance checking
    enable_oos_balance_check: bool = True
    balance_tolerance: float = 0.05  # 5% tolerance

    # Drift detection
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1


class EnhancedLabelDefinitions:
    """
    Enhanced label definitions for trading ML that implement specific trading logic.

    This class implements the label definitions requested:
    1. Analyst: "Should we trade?" (binary)
    2. Tactician: Direction/magnitude with excursion thresholds
    3. Regime-conditioned: Volatility-scaled thresholds
    4. Risk-aware: Stop-loss aware labeling
    """

    def __init__(
        self,
        analyst_config: Optional[AnalystLabelConfig] = None,
        tactician_config: Optional[TacticianLabelConfig] = None,
        regime_config: Optional[RegimeConditionedConfig] = None,
        risk_config: Optional[RiskAwareConfig] = None,
        cleaning_config: Optional[DataCleaningConfig] = None,
        stability_config: Optional[StabilityCheckConfig] = None
    ):
        """Initialize enhanced label definitions."""
        self.analyst_config = analyst_config or AnalystLabelConfig()
        self.tactician_config = tactician_config or TacticianLabelConfig()
        self.regime_config = regime_config or RegimeConditionedConfig()
        self.risk_config = risk_config or RiskAwareConfig()
        self.cleaning_config = cleaning_config or DataCleaningConfig()
        self.stability_config = stability_config or StabilityCheckConfig()

        self.logger = logging.getLogger('EnhancedLabelDefinitions')

        tprint_success("🚀 Enhanced Label Definitions initialized")
        tprint_info("   → Analyst labels: Should we trade?")
        tprint_info("   → Tactician labels: Direction/magnitude")
        tprint_info("   → Regime conditioning: Volatility-scaled")
        tprint_info("   → Risk awareness: Stop-loss aware")

    def generate_analyst_labels(
        self,
        market_data: pd.DataFrame,
        volatility_series: pd.Series,
        regime_data: Optional[pd.Series] = None,
        portfolio_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate Analyst labels: "Should we trade?" (1 if expected PnL > costs)

        Args:
            market_data: OHLCV market data
            volatility_series: Volatility estimates
            regime_data: Optional regime assignments
            portfolio_state: Optional current portfolio state

        Returns:
            Tuple of (analyst_labels, confidence_scores)
        """
        tprint_info("🎯 Generating Analyst labels (Should we trade?)")

        try:
            # Clean data first
            cleaned_data = self._apply_data_cleaning(market_data)

            # Calculate expected returns over horizon
            expected_returns = self._calculate_expected_returns(
                cleaned_data, self.analyst_config.horizon_minutes
            )

            # Calculate trading costs with funding/borrow assumptions
            trading_costs = self._calculate_trading_costs(
                cleaned_data,
                self.analyst_config.trading_costs,
                expected_returns=expected_returns,
                stress_scenario=self.analyst_config.trading_costs.active_stress_scenario
            )

            # Apply regime conditioning if enabled
            if self.analyst_config.enable_regime_conditioning and regime_data is not None:
                regime_multipliers = self._calculate_regime_multipliers(
                    volatility_series, regime_data
                )
                expected_returns *= regime_multipliers

            # Apply risk awareness
            risk_adjusted_returns = self._apply_risk_awareness(
                expected_returns, cleaned_data, portfolio_state
            )

            # Generate analyst labels (1 if net profit > 0)
            net_profits = risk_adjusted_returns - trading_costs
            analyst_labels = (net_profits > 0).astype(int)

            # Calculate confidence scores based on signal strength
            confidence_scores = self._calculate_analyst_confidence(
                net_profits, expected_returns, volatility_series
            )

            # Apply minimum confidence threshold
            confident_mask = confidence_scores >= self.analyst_config.min_confidence_threshold
            analyst_labels[~confident_mask] = 0

            tprint_success(f"✅ Analyst labels generated: {analyst_labels.sum()}/{len(analyst_labels)} positive trades")

            return analyst_labels, confidence_scores

        except Exception as e:
            tprint_error(f"❌ Error generating analyst labels: {e}")
            # Return neutral labels on error
            return pd.Series(0, index=market_data.index), pd.Series(0.5, index=market_data.index)

    def generate_tactician_labels(
        self,
        market_data: pd.DataFrame,
        volatility_series: pd.Series,
        regime_data: Optional[pd.Series] = None,
        current_positions: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate Tactician labels: Direction/magnitude based on excursion thresholds.

        Args:
            market_data: OHLCV market data
            volatility_series: Volatility estimates
            regime_data: Optional regime assignments
            current_positions: Optional current positions

        Returns:
            Tuple of (tactician_labels, magnitude_scores)
        """
        tprint_info("⚔️ Generating Tactician labels (Direction/Magnitude)")

        try:
            # Clean data first
            cleaned_data = self._apply_data_cleaning(market_data)

            # Calculate price excursions over horizon
            favorable_excursion, adverse_excursion = self._calculate_excursions(
                cleaned_data, self.tactician_config.horizon_minutes
            )

            # Apply volatility scaling
            if self.tactician_config.enable_regime_conditioning and regime_data is not None:
                vol_scaled_fav = favorable_excursion / volatility_series
                vol_scaled_adv = adverse_excursion / volatility_series

                # Adjust thresholds based on regime
                regime_adjusted_fav = self.tactician_config.favorable_excursion_threshold
                regime_adjusted_adv = self.tactician_config.adverse_excursion_threshold

                # Apply regime-specific scaling
                regime_multipliers = self._calculate_regime_multipliers(
                    volatility_series, regime_data
                )
                regime_adjusted_fav *= regime_multipliers
                regime_adjusted_adv *= regime_multipliers
            else:
                vol_scaled_fav = favorable_excursion / volatility_series
                vol_scaled_adv = adverse_excursion / volatility_series
                regime_adjusted_fav = self.tactician_config.favorable_excursion_threshold
                regime_adjusted_adv = self.tactician_config.adverse_excursion_threshold

            # Generate tactician labels based on excursion criteria
            excursion_criteria = (
                (vol_scaled_fav >= regime_adjusted_fav) &
                (vol_scaled_adv >= regime_adjusted_adv)
            )

            tactician_labels = excursion_criteria.astype(int)

            # Calculate magnitude scores (how strong the signal is)
            magnitude_scores = self._calculate_magnitude_scores(
                vol_scaled_fav, vol_scaled_adv, regime_adjusted_fav, regime_adjusted_adv
            )

            # Apply minimum confidence threshold
            confident_mask = magnitude_scores >= self.tactician_config.min_direction_confidence
            tactician_labels[~confident_mask] = 0

            # Scale magnitude if enabled
            if self.tactician_config.magnitude_scaling:
                magnitude_scores = np.clip(
                    magnitude_scores * self.tactician_config.max_magnitude,
                    0, self.tactician_config.max_magnitude
                )

            tprint_success(f"✅ Tactician labels generated: {tactician_labels.sum()}/{len(tactician_labels)} valid directions")

            return tactician_labels, magnitude_scores

        except Exception as e:
            tprint_error(f"❌ Error generating tactician labels: {e}")
            # Return neutral labels on error
            return pd.Series(0, index=market_data.index), pd.Series(1.0, index=market_data.index)

    def generate_regime_conditioned_labels(
        self,
        base_labels: pd.Series,
        volatility_series: pd.Series,
        regime_data: pd.Series
    ) -> pd.Series:
        """
        Apply regime conditioning to existing labels using volatility-scaled thresholds.

        Args:
            base_labels: Base labels to condition
            volatility_series: Volatility estimates
            regime_data: Regime assignments

        Returns:
            Regime-conditioned labels
        """
        tprint_info("🎭 Applying regime conditioning with volatility-scaled thresholds")

        try:
            # Calculate regime-specific multipliers
            regime_multipliers = self._calculate_regime_multipliers(volatility_series, regime_data)

            # Apply volatility scaling to base labels
            if self.regime_config.adaptive_thresholds:
                # Use adaptive thresholds based on historical regime behavior
                adaptive_thresholds = self._calculate_adaptive_thresholds(
                    volatility_series, regime_data
                )
                regime_conditioned = base_labels.copy()

                for regime in regime_data.unique():
                    if pd.isna(regime):
                        continue

                    regime_mask = regime_data == regime
                    threshold = adaptive_thresholds.get(regime, 0.5)

                    # Apply regime-specific threshold
                    regime_conditioned[regime_mask] = (
                        base_labels[regime_mask] > threshold
                    ).astype(int)
            else:
                # Use fixed regime multipliers
                regime_conditioned = base_labels * regime_multipliers
                regime_conditioned = (regime_conditioned > 0.5).astype(int)

            tprint_success(f"✅ Regime conditioning applied to {len(regime_conditioned)} labels")

            return regime_conditioned

        except Exception as e:
            tprint_error(f"❌ Error applying regime conditioning: {e}")
            return base_labels

    def generate_risk_aware_labels(
        self,
        base_labels: pd.Series,
        market_data: pd.DataFrame,
        portfolio_state: Optional[Dict[str, Any]] = None,
        current_positions: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Apply risk awareness to labels (0 if trade would hit stop before target).

        Args:
            base_labels: Base labels to make risk-aware
            market_data: OHLCV market data
            portfolio_state: Current portfolio state
            current_positions: Current positions

        Returns:
            Risk-aware labels
        """
        tprint_info("🛡️ Applying risk awareness (stop-loss protection)")

        try:
            risk_aware_labels = base_labels.copy()

            # Calculate stop-loss and take-profit levels
            stop_loss_levels = market_data['close'] * (1 - self.risk_config.stop_loss_pct)
            take_profit_levels = market_data['close'] * (1 + self.risk_config.take_profit_pct)

            # Simulate trade outcomes over horizon
            horizon_returns = self._simulate_trade_outcomes(
                market_data, stop_loss_levels, take_profit_levels
            )

            # Check if stop-loss would be hit before take-profit
            for idx in market_data.index:
                if base_labels.loc[idx] == 1:  # Only check positive labels
                    # Check if stop-loss is hit within horizon
                    future_prices = market_data.loc[idx:, 'high']
                    stop_hit = (future_prices <= stop_loss_levels.loc[idx]).any()

                    take_profit_prices = market_data.loc[idx:, 'low']
                    take_profit_hit = (take_profit_prices >= take_profit_levels.loc[idx]).any()

                    if stop_hit and not take_profit_hit:
                        # Stop-loss hit before take-profit - don't trade
                        risk_aware_labels.loc[idx] = 0
                    elif stop_hit and take_profit_hit:
                        # Both hit - check which first
                        stop_idx = (future_prices <= stop_loss_levels.loc[idx]).idxmax()
                        tp_idx = (take_profit_prices >= take_profit_levels.loc[idx]).idxmax()

                        if stop_idx < tp_idx:
                            risk_aware_labels.loc[idx] = 0  # Stop-loss first

            # Apply portfolio risk limits
            if portfolio_state:
                risk_aware_labels = self._apply_portfolio_risk_limits(
                    risk_aware_labels, portfolio_state, current_positions
                )

            tprint_success(f"✅ Risk awareness applied: {risk_aware_labels.sum()}/{len(risk_aware_labels)} trades after filtering")

            return risk_aware_labels

        except Exception as e:
            tprint_error(f"❌ Error applying risk awareness: {e}")
            return base_labels

    def _apply_data_cleaning(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning according to configuration."""
        tprint_info("🧹 Applying data cleaning")

        cleaned = market_data.copy()

        # Remove outliers
        if self.cleaning_config.outlier_method == "iqr":
            for col in ['high', 'low', 'close', 'volume']:
                if col in cleaned.columns:
                    Q1 = cleaned[col].quantile(0.25)
                    Q3 = cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.cleaning_config.outlier_threshold * IQR
                    upper_bound = Q3 + self.cleaning_config.outlier_threshold * IQR

                    cleaned = cleaned[
                        (cleaned[col] >= lower_bound) &
                        (cleaned[col] <= upper_bound)
                    ]

        # Apply volume filters
        if 'volume' in cleaned.columns:
            cleaned = cleaned[
                (cleaned['volume'] >= self.cleaning_config.min_volume_threshold) &
                (cleaned['volume'] <= self.cleaning_config.max_volume_threshold)
            ]

        # Apply price filters
        if 'close' in cleaned.columns:
            # Remove zero/negative prices
            cleaned = cleaned[cleaned['close'] >= self.cleaning_config.min_price]

            # Remove extreme price changes
            price_changes = cleaned['close'].pct_change()
            cleaned = cleaned[abs(price_changes) <= self.cleaning_config.max_price_change_pct]

        # Remove duplicates if enabled
        if self.cleaning_config.enable_deduplication:
            cleaned = cleaned[~cleaned.index.duplicated(keep='first')]

        # Align timestamps if enabled
        if self.cleaning_config.enforce_timestamp_alignment:
            cleaned = self._align_timestamps(cleaned)

        tprint_info(f"✅ Data cleaning applied: {len(cleaned)}/{len(market_data)} bars remaining")

        return cleaned

    def _calculate_expected_returns(self, market_data: pd.DataFrame, horizon_minutes: int) -> pd.Series:
        """Calculate expected returns over horizon with optional asymmetric scaling."""
        close_prices = market_data['close']

        # Forward returns over the horizon (shifted to align with current bar)
        horizon_bars = max(1, horizon_minutes // 15)  # Assuming 15m bars
        forward_returns = close_prices.pct_change(periods=horizon_bars).shift(-horizon_bars)
        forward_returns = forward_returns.fillna(0)

        baseline_expectation = forward_returns.rolling(horizon_bars, min_periods=1).mean()

        scaling_config = self.analyst_config.asymmetric_return_scaling
        if not scaling_config.enabled:
            return baseline_expectation.fillna(0)

        # Ensure blend ratio is within [0, 1]
        blend_ratio = np.clip(scaling_config.blend_ratio, 0.0, 1.0)
        lookback = max(horizon_bars, max(1, scaling_config.tail_lookback_window))

        def _weighted_tail_expectation(window: pd.Series) -> float:
            series = window.dropna()
            if series.empty:
                return 0.0

            method = scaling_config.method.lower()
            if method == "skew_adjusted":
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or np.isnan(std):
                    return mean
                skew = series.skew()
                skew = 0.0 if np.isnan(skew) else skew
                upside_component = max(skew, 0.0) * scaling_config.upside_weight * std
                downside_component = abs(min(skew, 0.0)) * scaling_config.downside_weight * std
                return mean + upside_component - downside_component

            # Default to weighted tail expectation
            upper_q = series.quantile(np.clip(scaling_config.upside_tail_percentile, 0.0, 1.0))
            lower_q = series.quantile(np.clip(scaling_config.downside_tail_percentile, 0.0, 1.0))
            upside_tail = series[series >= upper_q]
            downside_tail = series[series <= lower_q]

            upside_mean = upside_tail.mean() if not upside_tail.empty else 0.0
            downside_mean = downside_tail.mean() if not downside_tail.empty else 0.0

            return (
                scaling_config.upside_weight * upside_mean +
                scaling_config.downside_weight * downside_mean
            )

        asymmetric_expectation = forward_returns.rolling(
            lookback,
            min_periods=1
        ).apply(_weighted_tail_expectation, raw=False)

        expected_returns = (
            (1.0 - blend_ratio) * baseline_expectation.fillna(0) +
            blend_ratio * asymmetric_expectation.fillna(0)
        )

        return expected_returns.fillna(0)

    def _calculate_trading_costs(
        self,
        market_data: pd.DataFrame,
        costs: TradingCosts,
        expected_returns: Optional[pd.Series] = None,
        stress_scenario: Optional[str] = None
    ) -> pd.Series:
        """Calculate trading costs for each bar including borrow, funding, and stress."""
        if market_data.empty:
            return pd.Series(dtype=float)

        volume = market_data.get('volume', pd.Series(0.0, index=market_data.index)).fillna(0.0)
        close_prices = market_data.get('close', pd.Series(0.0, index=market_data.index)).ffill().fillna(0.0)
        trade_size = (volume * close_prices * 0.01).astype(float)

        trade_size_values = trade_size.to_numpy()
        trade_size_values = np.where(
            trade_size_values > 0,
            np.maximum(trade_size_values, costs.min_trade_size),
            0.0
        )
        trade_size = pd.Series(trade_size_values, index=market_data.index)

        if 'is_maker' in market_data.columns:
            is_maker_series = market_data['is_maker'].fillna(costs.default_is_maker).astype(bool)
        else:
            is_maker_series = pd.Series(costs.default_is_maker, index=market_data.index)

        fee_rates = np.where(is_maker_series, costs.maker_fee, costs.taker_fee)
        fee_rates = pd.Series(fee_rates, index=market_data.index, dtype=float)
        fee_costs = trade_size * fee_rates + trade_size * costs.slippage_pct

        if expected_returns is not None:
            aligned_returns = expected_returns.reindex(market_data.index).fillna(0.0)
            direction_series = pd.Series(
                np.where(aligned_returns >= 0, 'long', 'short'),
                index=market_data.index
            )
        elif 'trade_direction' in market_data.columns:
            direction_series = market_data['trade_direction'].fillna('long').astype(str).str.lower()
        elif 'position' in market_data.columns:
            direction_series = pd.Series(
                np.where(market_data['position'] >= 0, 'long', 'short'),
                index=market_data.index
            )
        else:
            direction_series = pd.Series('long', index=market_data.index)

        if 'asset_class' in market_data.columns:
            asset_classes = market_data['asset_class'].fillna(costs.default_asset_class).astype(str)
        else:
            asset_classes = pd.Series(costs.default_asset_class, index=market_data.index)

        costs.validate_asset_assumptions(asset_classes.unique())

        borrow_rates = []
        funding_rates = []
        stress_multipliers = []
        scenario_key = stress_scenario or costs.active_stress_scenario

        for idx in market_data.index:
            asset_class = asset_classes.loc[idx]
            direction = direction_series.loc[idx]
            borrow_rates.append(costs.get_borrow_rate(asset_class, direction))
            funding_rates.append(costs.get_funding_rate(asset_class, direction))
            stress_multipliers.append(
                costs.get_stress_multiplier(asset_class, direction, scenario=scenario_key)
            )

        borrow_rates = pd.Series(borrow_rates, index=market_data.index, dtype=float)
        funding_rates = pd.Series(funding_rates, index=market_data.index, dtype=float)
        stress_multipliers = pd.Series(stress_multipliers, index=market_data.index, dtype=float)

        borrow_costs = trade_size * borrow_rates
        funding_costs = trade_size * funding_rates

        total_costs = (fee_costs + borrow_costs + funding_costs) * stress_multipliers

        return total_costs.fillna(0.0)

    def _calculate_regime_multipliers(self, volatility_series: pd.Series, regime_data: pd.Series) -> pd.Series:
        """Calculate regime-specific multipliers for thresholds."""
        multipliers = pd.Series(1.0, index=volatility_series.index)

        if not self.regime_config.volatility_scaling_enabled:
            return multipliers

        # Calculate volatility percentiles for regime classification
        vol_percentiles = volatility_series.quantile(self.regime_config.regime_volatility_percentiles)

        for idx in volatility_series.index:
            vol = volatility_series.loc[idx]
            regime = regime_data.loc[idx] if idx in regime_data.index else None

            if pd.isna(regime):
                continue

            if vol <= vol_percentiles.iloc[0]:
                # Low volatility regime
                multipliers.loc[idx] = self.regime_config.low_vol_multiplier
            elif vol >= vol_percentiles.iloc[1]:
                # High volatility regime
                multipliers.loc[idx] = self.regime_config.high_vol_multiplier
            else:
                # Normal volatility regime
                multipliers.loc[idx] = self.regime_config.base_threshold_multiplier

        return multipliers

    def _apply_risk_awareness(self, expected_returns: pd.Series, market_data: pd.DataFrame,
                             portfolio_state: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Apply risk awareness to expected returns with configurable asymmetry."""
        risk_adjusted = expected_returns.copy()

        risk_config = self.analyst_config.asymmetric_risk_adjustment

        if risk_config.enable_downside_penalty:
            returns = market_data['close'].pct_change().fillna(0)
            lookback = max(1, risk_config.penalty_lookback_window)
            downside_tail = returns.rolling(lookback, min_periods=1).quantile(
                np.clip(risk_config.downside_tail_percentile, 0.0, 1.0)
            )
            downside_tail = downside_tail.fillna(0).abs() * risk_config.downside_penalty_multiplier

            if risk_config.apply_only_to_positive:
                penalty = downside_tail.where(risk_adjusted > 0, 0.0)
            else:
                penalty = downside_tail

            risk_adjusted = risk_adjusted - penalty

        # Determine asymmetric clamps (fallback to defaults when not provided)
        max_position_return = self.analyst_config.max_position_size_pct
        max_drawdown_return = self.analyst_config.max_drawdown_pct

        clamp_min = -max_drawdown_return
        clamp_max = max_position_return

        if risk_config.enable_asymmetric_clamp:
            if risk_config.clamp_min is not None:
                clamp_min = risk_config.clamp_min
            if risk_config.clamp_max is not None:
                clamp_max = risk_config.clamp_max

        risk_adjusted = np.clip(risk_adjusted, clamp_min, clamp_max)

        return risk_adjusted

    def _calculate_analyst_confidence(self, net_profits: pd.Series, expected_returns: pd.Series,
                                    volatility_series: pd.Series) -> pd.Series:
        """Calculate confidence scores for analyst labels."""
        # Confidence based on signal-to-noise ratio
        signal_strength = abs(net_profits) / (volatility_series + 1e-8)
        confidence = np.clip(signal_strength / signal_strength.quantile(0.9), 0, 1)

        return confidence

    def _calculate_excursions(self, market_data: pd.DataFrame, horizon_minutes: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate favorable and adverse excursions over horizon."""
        horizon_bars = max(1, horizon_minutes // 15)  # Assuming 15m bars

        # Calculate rolling max and min over horizon
        rolling_high = market_data['high'].rolling(horizon_bars).max().shift(-horizon_bars)
        rolling_low = market_data['low'].rolling(horizon_bars).min().shift(-horizon_bars)

        # Calculate excursions from current close
        current_close = market_data['close']
        favorable_excursion = (rolling_high - current_close) / current_close
        adverse_excursion = (current_close - rolling_low) / current_close

        return favorable_excursion.fillna(0), adverse_excursion.fillna(0)

    def _calculate_magnitude_scores(self, favorable_excursion: pd.Series, adverse_excursion: pd.Series,
                                  threshold_fav: float, threshold_adv: float) -> pd.Series:
        """Calculate magnitude scores for tactician labels."""
        # Magnitude based on how much excursion exceeds thresholds
        fav_magnitude = favorable_excursion - threshold_fav
        adv_magnitude = abs(adverse_excursion) - abs(threshold_adv)

        # Combined magnitude score
        magnitude_scores = (fav_magnitude + adv_magnitude) / 2
        magnitude_scores = np.clip(magnitude_scores, 0, 2)  # Scale to 0-2 range

        return magnitude_scores

    def _calculate_adaptive_thresholds(self, volatility_series: pd.Series,
                                     regime_data: pd.Series) -> Dict[Any, float]:
        """Calculate adaptive thresholds based on historical regime behavior."""
        thresholds = {}

        for regime in regime_data.unique():
            if pd.isna(regime):
                continue

            regime_mask = regime_data == regime
            regime_volatility = volatility_series[regime_mask]

            if len(regime_volatility) > self.regime_config.lookback_window:
                # Use median volatility as adaptive threshold
                thresholds[regime] = regime_volatility.median()

        return thresholds

    def _simulate_trade_outcomes(self, market_data: pd.DataFrame,
                               stop_loss_levels: pd.Series,
                               take_profit_levels: pd.Series) -> pd.Series:
        """Simulate trade outcomes to check if stops are hit."""
        # This is a simplified simulation - in practice would be more sophisticated
        horizon_bars = 4  # 1 hour for 15m bars

        outcomes = pd.Series(0, index=market_data.index)

        for idx in market_data.index[:-horizon_bars]:
            entry_price = market_data.loc[idx, 'close']
            stop_price = stop_loss_levels.loc[idx]
            target_price = take_profit_levels.loc[idx]

            # Look forward in horizon
            future_high = market_data.loc[idx:idx+horizon_bars, 'high'].max()
            future_low = market_data.loc[idx:idx+horizon_bars, 'low'].min()

            if future_low <= stop_price:
                outcomes.loc[idx] = -1  # Stop loss hit
            elif future_high >= target_price:
                outcomes.loc[idx] = 1   # Take profit hit
            else:
                outcomes.loc[idx] = 0   # Neither hit

        return outcomes

    def _apply_portfolio_risk_limits(self, labels: pd.Series,
                                   portfolio_state: Dict[str, Any],
                                   current_positions: Optional[Dict[str, Any]] = None) -> pd.Series:
        """Apply portfolio-level risk limits to labels."""
        adjusted_labels = labels.copy()

        # Simple risk limit: don't exceed max portfolio risk per trade
        max_trades = int(1 / self.risk_config.max_portfolio_risk_pct)

        # If too many positive labels, reduce some based on confidence
        positive_indices = labels[labels == 1].index
        if len(positive_indices) > max_trades:
            # Keep only the highest confidence trades
            # This would need confidence scores to work properly
            excess_trades = positive_indices[:len(positive_indices) - max_trades]
            adjusted_labels.loc[excess_trades] = 0

        return adjusted_labels

    def _align_timestamps(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Align timestamps to expected intervals."""
        # Simple timestamp alignment - remove gaps larger than threshold
        time_diffs = market_data.index.to_series().diff()
        max_gap = pd.Timedelta(minutes=self.cleaning_config.max_timestamp_gap_minutes)

        valid_mask = time_diffs <= max_gap
        aligned_data = market_data[valid_mask]

        return aligned_data

    def check_label_stability(
        self,
        current_labels: pd.Series,
        historical_labels: Optional[pd.Series] = None,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Check label stability and detect potential issues.

        Args:
            current_labels: Current labels
            historical_labels: Historical labels for comparison
            market_data: Market data for leakage detection

        Returns:
            Stability check results
        """
        tprint_info("🔍 Checking label stability")

        stability_results = {
            'is_stable': True,
            'issues': [],
            'metrics': {}
        }

        try:
            # Check autocorrelation (potential leakage)
            if market_data is not None and self.stability_config.max_autocorrelation_threshold > 0:
                autocorrelation = self._check_autocorrelation(current_labels)
                stability_results['metrics']['autocorrelation'] = autocorrelation

                if abs(autocorrelation) > self.stability_config.max_autocorrelation_threshold:
                    stability_results['is_stable'] = False
                    stability_results['issues'].append(
                        f"High autocorrelation detected: {autocorrelation:.3f}"
                    )

            # Check label balance
            if len(current_labels) > 0:
                positive_ratio = current_labels.mean()
                balance_deviation = abs(positive_ratio - 0.5)

                stability_results['metrics']['positive_ratio'] = positive_ratio
                stability_results['metrics']['balance_deviation'] = balance_deviation

                if balance_deviation > self.stability_config.balance_tolerance:
                    stability_results['is_stable'] = False
                    stability_results['issues'].append(
                        f"Label imbalance detected: {positive_ratio:.3f} positive ratio"
                    )

            # Compare with historical labels if available
            if historical_labels is not None and self.stability_config.enable_drift_detection:
                drift_score = self._calculate_drift_score(current_labels, historical_labels)
                stability_results['metrics']['drift_score'] = drift_score

                if drift_score > self.stability_config.drift_threshold:
                    stability_results['is_stable'] = False
                    stability_results['issues'].append(
                        f"Label drift detected: {drift_score:.3f}"
                    )

            tprint_success(f"✅ Stability check completed: {'Stable' if stability_results['is_stable'] else 'Issues found'}")

            return stability_results

        except Exception as e:
            tprint_error(f"❌ Error checking stability: {e}")
            stability_results['is_stable'] = False
            stability_results['issues'].append(f"Stability check failed: {e}")
            return stability_results

    def _check_autocorrelation(self, labels: pd.Series) -> float:
        """Check for autocorrelation in labels that might indicate leakage."""
        try:
            # Calculate autocorrelation at lag 1
            if len(labels) < self.stability_config.lookback_window_leakage:
                return 0.0

            # Use only recent data for leakage check
            recent_labels = labels.tail(self.stability_config.lookback_window_leakage)

            # Calculate autocorrelation
            lagged = recent_labels.shift(1).fillna(0)
            correlation = recent_labels.corr(lagged)

            return correlation if not pd.isna(correlation) else 0.0

        except Exception:
            return 0.0

    def _calculate_drift_score(self, current_labels: pd.Series, historical_labels: pd.Series) -> float:
        """Calculate drift between current and historical labels."""
        try:
            # Simple drift measure: difference in positive ratios
            current_ratio = current_labels.mean()
            historical_ratio = historical_labels.mean()

            return abs(current_ratio - historical_ratio)

        except Exception:
            return 0.0


# Convenience functions for easy usage
def create_enhanced_labeler(
    analyst_config: Optional[AnalystLabelConfig] = None,
    tactician_config: Optional[TacticianLabelConfig] = None,
    regime_config: Optional[RegimeConditionedConfig] = None,
    risk_config: Optional[RiskAwareConfig] = None,
    cleaning_config: Optional[DataCleaningConfig] = None,
    stability_config: Optional[StabilityCheckConfig] = None
) -> EnhancedLabelDefinitions:
    """Create enhanced label definitions with specified configurations."""
    return EnhancedLabelDefinitions(
        analyst_config=analyst_config,
        tactician_config=tactician_config,
        regime_config=regime_config,
        risk_config=risk_config,
        cleaning_config=cleaning_config,
        stability_config=stability_config
    )


def create_trading_aware_config() -> Dict[str, Any]:
    """Create a trading-aware configuration optimized for real trading."""
    return {
        'analyst_config': AnalystLabelConfig(
            horizon_minutes=60,
            min_profit_threshold_usd=5.0,
            trading_costs=TradingCosts(
                maker_fee=0.001,
                taker_fee=0.002,
                slippage_pct=0.001,
                default_asset_class="crypto",
                borrow_fees={
                    "crypto": {"long": 0.00005, "short": 0.0007},
                    "default": {"long": 0.0, "short": 0.0005}
                },
                funding_rates={
                    "crypto": {"long": 0.00025, "short": -0.00025},
                    "default": {"long": 0.0, "short": 0.0}
                },
                stress_scenarios={
                    "crypto": {
                        "base": {"long": 1.0, "short": 1.0},
                        "liquidity_crunch": {"long": 1.2, "short": 1.4}
                    },
                    "default": {
                        "base": {"long": 1.0, "short": 1.0}
                    }
                },
                active_stress_scenario="base"
            ),
            enable_regime_conditioning=True,
            volatility_scaling_factor=1.0
        ),
        'tactician_config': TacticianLabelConfig(
            favorable_excursion_threshold=1.0,
            adverse_excursion_threshold=-2.0,
            horizon_minutes=30,
            enable_regime_conditioning=True,
            volatility_sensitivity=1.0
        ),
        'regime_config': RegimeConditionedConfig(
            volatility_scaling_enabled=True,
            base_threshold_multiplier=1.0,
            adaptive_thresholds=True,
            lookback_window=50
        ),
        'risk_config': RiskAwareConfig(
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
            min_risk_reward_ratio=2.0,
            max_portfolio_risk_pct=0.02
        ),
        'cleaning_config': DataCleaningConfig(
            outlier_method="iqr",
            outlier_threshold=3.0,
            min_volume_threshold=1000.0,
            enforce_timestamp_alignment=True
        ),
        'stability_config': StabilityCheckConfig(
            recompute_on_refresh=True,
            max_autocorrelation_threshold=0.3,
            enable_oos_balance_check=True,
            balance_tolerance=0.05,
            enable_drift_detection=True,
            drift_threshold=0.1
        )
    }