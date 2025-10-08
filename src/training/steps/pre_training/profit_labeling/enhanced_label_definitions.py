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
import hashlib
import time

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_correlation
)
from src.utils.math_validation import MathValidation
from collections import defaultdict

# Import matrix operations and hardware optimization
from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor, HardwareConfig
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer


class LabelDefinitionType(Enum):
    """Types of label definitions."""
    ANALYST = "analyst"           # Should we trade? (0/1)
    TACTICIAN = "tactician"       # Direction/magnitude (0/1 with thresholds)
    REGIME_CONDITIONED = "regime_conditioned"  # Volatility-scaled
    RISK_AWARE = "risk_aware"     # Stop-loss aware


@dataclass
class TradingCosts:
    """Trading costs configuration."""
    maker_fee: float = 0.0001    # 0.01% maker fee
    taker_fee: float = 0.00025   # 0.025% taker fee (market orders)
    slippage_pct: float = 0.00025  # 0.025% slippage estimate
    min_trade_size: float = 10.0  # Minimum trade size in USD
    # Roundtrip cost: (0.025% + 0.025%) × 2 = 0.1% total
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

    # Bar duration in minutes (configurable instead of hardcoded 15)
    bar_duration_minutes: int = 15

    # Profitability thresholds
    min_profit_threshold_usd: float = 5.0
    min_profit_threshold_pct: float = 0.001  # 0.1%

    # Trading costs
    trading_costs: TradingCosts = field(default_factory=TradingCosts)

    # Confidence thresholds
    min_confidence_threshold: float = 0.4  # Relaxed from 0.6 to allow more labels

    # Data quality filters
    min_volume_threshold: float = 1000.0
    max_spread_pct: float = 0.01  # 1%

    # Risk management
    max_position_size_pct: float = 0.05  # 5% of portfolio
    max_drawdown_pct: float = 0.02      # 2% max drawdown

    # Capacity management
    enforce_capacity_limits: bool = True
    min_holding_minutes: int = 0
    max_turnover_per_day: Optional[float] = None
    capacity_violation_action: str = "scale_confidence"  # "scale_confidence" or "zero_out"
    capacity_scaling_factor: float = 0.5
    impact_cost_per_unit_turnover: float = 0.0
    impact_penalty_exponent: float = 1.0
    max_impact_cost_pct: Optional[float] = None

    # Regime conditioning
    enable_regime_conditioning: bool = True
    volatility_scaling_factor: float = 1.0
    
    # Trading direction settings
    enable_long_positions: bool = True   # Include long opportunities (buy when expecting price increase)
    enable_short_positions: bool = False  # Include short opportunities (sell when expecting price decrease)

    # Performance optimization settings
    enable_caching: bool = True
    cache_duration_minutes: int = 60
    enable_hardware_optimization: bool = True
    enable_vectorized_operations: bool = True

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
    regime_volatility_percentiles: Tuple[float, float] = (0.25, 0.75)  # 25th and 75th percentiles


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
        self._last_execution_metadata: Dict[str, Any] = {}
        self._latest_analyst_diagnostics: Dict[str, Any] = {}

        # Initialize performance optimization tools
        self._initialize_optimization_tools()

        # Initialize caching for intermediate results
        self._calculation_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}

        tprint_success("🚀 Enhanced Label Definitions initialized")
        tprint_info("   → Analyst labels: Should we trade?")
        tprint_info("   → Tactician labels: Direction/magnitude")
        tprint_info("   → Regime conditioning: Volatility-scaled")
        tprint_info("   → Risk awareness: Stop-loss aware")
        tprint_info("   → Performance optimization: Enabled")
        tprint_info("   → Hardware acceleration: Available" if self.matrix_ops else "   → Hardware acceleration: Not available")

    def _initialize_optimization_tools(self):
        """Initialize performance optimization tools."""
        try:
            # Initialize matrix operations
            self.matrix_ops = get_unified_matrix_operations()
            self.hardware_processor = HardwareOptimizedMatrixProcessor(HardwareConfig())

            # Initialize hardware optimizers if available
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_info("   → Hardware optimizers: M1/M2/M3 optimizations enabled")
            except Exception as e:
                tprint_warning(f"   → Hardware optimizers: Not available ({e})")
                self.memory_optimizer = None
                self.cpu_optimizer = None

            tprint_info("   → Matrix operations: Available")
            tprint_info("   → Vectorized processing: Available")

        except Exception as e:
            tprint_warning(f"⚠️ Performance optimization tools initialization failed: {e}")
            self.matrix_ops = None
            self.hardware_processor = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    def _generate_cache_key(self, data_hash: str, config_hash: str, calculation_type: str) -> str:
        """Generate a cache key for intermediate calculations."""
        return f"{calculation_type}_{data_hash}_{config_hash}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if not self.analyst_config.enable_caching:
            return False

        if cache_key not in self._cache_timestamps:
            return False

        cache_age = time.time() - self._cache_timestamps[cache_key]
        return cache_age < (self.analyst_config.cache_duration_minutes * 60)

    def _cache_result(self, cache_key: str, result: Any):
        """Cache an intermediate calculation result."""
        if self.analyst_config.enable_caching:
            self._calculation_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Retrieve a cached result if valid."""
        if self._is_cache_valid(cache_key):
            return self._calculation_cache.get(cache_key)
        return None

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

            # Calculate execution-aware expected returns over horizon
            execution_context = self._build_execution_context(
                cleaned_data, self.analyst_config.horizon_minutes
            )
            expected_returns = self._calculate_expected_returns(
                cleaned_data,
                self.analyst_config.horizon_minutes,
                entry_prices=execution_context['entry_prices'],
                exit_prices=execution_context['exit_prices']
            )

            # Calculate trading costs using delayed execution assumptions
            trading_costs = self._calculate_trading_costs(
                cleaned_data,
                self.analyst_config.trading_costs,
                entry_prices=execution_context['entry_prices'],
                exit_prices=execution_context['exit_prices']
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

            # Generate analyst labels based on directional settings
            # Long opportunity: positive returns > costs
            # Short opportunity: negative returns magnitude > costs (i.e., risk_adjusted_returns < -costs)
            net_profit_pct_long = risk_adjusted_returns - trading_costs
            net_profit_pct_short = -risk_adjusted_returns - trading_costs  # Profit from shorting
            
            # Build label mask based on enabled directions
            long_signals = (net_profit_pct_long > 0) if self.analyst_config.enable_long_positions else pd.Series(False, index=risk_adjusted_returns.index)
            short_signals = (net_profit_pct_short > 0) if self.analyst_config.enable_short_positions else pd.Series(False, index=risk_adjusted_returns.index)
            
            # Label = 1 if any enabled direction is profitable
            analyst_labels = (long_signals | short_signals).astype(int)
            initial_positive = analyst_labels.sum()
            
            # Store which direction is more profitable for reference
            net_profit_pct = pd.Series(
                np.where(net_profit_pct_long > net_profit_pct_short, net_profit_pct_long, net_profit_pct_short),
                index=risk_adjusted_returns.index
            )
            
            tprint_info(f"📊 Initial profitable signals: {initial_positive}/{len(analyst_labels)} (long={long_signals.sum()}, short={short_signals.sum()})")

            invalid_entries = execution_context['entry_prices'].isna() | execution_context['exit_prices'].isna()
            if invalid_entries.any():
                # Only set values for indices that exist in analyst_labels
                common_indices = analyst_labels.index.intersection(invalid_entries.index)
                if len(common_indices) > 0:
                    before_invalid = analyst_labels.sum()
                    analyst_labels.loc[common_indices] = analyst_labels.loc[common_indices].where(~invalid_entries.loc[common_indices], 0)
                    net_profit_pct.loc[common_indices] = net_profit_pct.loc[common_indices].where(~invalid_entries.loc[common_indices], 0.0)
                    after_invalid = analyst_labels.sum()
                    tprint_info(f"📊 After invalid entry/exit filter: {after_invalid}/{len(analyst_labels)} (removed {before_invalid - after_invalid})")

            # Calculate confidence scores based on signal strength
            # Use percentage values for confidence calculation
            confidence_scores = self._calculate_analyst_confidence(
                net_profit_pct, risk_adjusted_returns, volatility_series
            )

            # Apply minimum confidence threshold
            before_confidence = analyst_labels.sum()
            confident_mask = confidence_scores >= self.analyst_config.min_confidence_threshold
            analyst_labels[~confident_mask] = 0
            after_confidence = analyst_labels.sum()
            tprint_info(f"📊 After confidence filter (≥{self.analyst_config.min_confidence_threshold}): {after_confidence}/{len(analyst_labels)} (removed {before_confidence - after_confidence})")

            # Apply capacity and turnover constraints
            before_capacity = analyst_labels.sum()
            (
                analyst_labels,
                confidence_scores,
                capacity_diagnostics
            ) = self._apply_capacity_constraints(
                analyst_labels,
                confidence_scores,
                cleaned_data.index,
                net_profit_pct
            )
            self._latest_analyst_diagnostics = capacity_diagnostics
            after_capacity = analyst_labels.sum()
            tprint_info(f"📊 After capacity constraints: {after_capacity}/{len(analyst_labels)} (removed {before_capacity - after_capacity})")

            tprint_success(
                "✅ Analyst labels generated: "
                f"{analyst_labels.sum()}/{len(analyst_labels)} positive trades"
            )
            tprint_info(
                "   → Capacity score: "
                f"{capacity_diagnostics['capacity_score']:.2f}, "
                f"turnover: {capacity_diagnostics['realized_turnover']:.2f}"
            )
            if capacity_diagnostics.get('violations_flagged'):
                tprint_warning("   ⚠️ Capacity or impact limits triggered; labels adjusted")

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

            # Calculate stop-loss and take-profit levels using delayed execution prices
            execution_context = self._build_execution_context(
                market_data, self.analyst_config.horizon_minutes
            )
            entry_prices = execution_context['entry_prices']
            stop_loss_levels = entry_prices * (1 - self.risk_config.stop_loss_pct)
            take_profit_levels = entry_prices * (1 + self.risk_config.take_profit_pct)

            # Simulate trade outcomes with delayed execution assumptions
            trade_outcomes = self._simulate_trade_outcomes(
                market_data,
                stop_loss_levels,
                take_profit_levels,
                entry_prices=entry_prices,
                horizon_bars=execution_context['horizon_bars']
            )

            # Remove trades that cannot be executed due to missing forward data
            missing_execution = entry_prices.isna() | execution_context['exit_prices'].isna()
            if missing_execution.any():
                risk_aware_labels.loc[missing_execution] = 0

            # Disable trades where simulated stop-loss triggers before target
            for idx in market_data.index:
                if base_labels.loc[idx] == 1:
                    if trade_outcomes.loc[idx] == -1:
                        risk_aware_labels.loc[idx] = 0

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

    def get_latest_analyst_diagnostics(self) -> Dict[str, Any]:
        """Return the most recent analyst label capacity diagnostics."""
        return dict(self._latest_analyst_diagnostics)

    def _apply_capacity_constraints(
        self,
        analyst_labels: pd.Series,
        confidence_scores: pd.Series,
        index: pd.Index,
        net_profits: pd.Series
    ) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
        """Apply capacity, turnover, and holding period constraints."""

        config = self.analyst_config
        diagnostics: Dict[str, Any] = {
            'enforce_capacity_limits': config.enforce_capacity_limits,
            'min_holding_minutes': config.min_holding_minutes,
            'max_turnover_per_day': config.max_turnover_per_day,
            'capacity_violation_action': config.capacity_violation_action,
            'capacity_scaling_factor': config.capacity_scaling_factor,
            'min_holding_violations': 0,
            'turnover_violations': 0,
            'impact_violations': 0,
            'violating_timestamps': [],
            'scaled_timestamps': []
        }

        if analyst_labels.empty:
            diagnostics.update({
                'realized_turnover': 0.0,
                'daily_turnover': {},
                'capacity_score': 1.0,
                'violations_flagged': False,
                'capacity_utilization': 0.0,
                'impact_cost': 0.0,
                'trading_days_evaluated': 0
            })
            return analyst_labels, confidence_scores, diagnostics

        labels_adjusted = analyst_labels.copy().astype(int)
        confidence_adjusted = confidence_scores.reindex(labels_adjusted.index)
        if confidence_adjusted.isnull().any():
            confidence_adjusted = confidence_adjusted.ffill().bfill().fillna(0.0)

        if (
            config.enforce_capacity_limits and
            config.min_holding_minutes > 0 and
            isinstance(index, pd.DatetimeIndex) and
            len(labels_adjusted) > 1
        ):
            min_hold_delta = pd.Timedelta(minutes=config.min_holding_minutes)
            last_change_time = index[0]

            for i in range(1, len(labels_adjusted)):
                ts = index[i]
                prev_value = labels_adjusted.iat[i - 1]
                proposed_value = labels_adjusted.iat[i]

                if proposed_value != prev_value:
                    elapsed = ts - last_change_time
                    if elapsed < min_hold_delta:
                        labels_adjusted.iat[i] = prev_value
                        confidence_adjusted.iat[i] *= np.clip(config.capacity_scaling_factor, 0.0, 1.0)
                        diagnostics['min_holding_violations'] += 1
                        diagnostics['scaled_timestamps'].append(ts)
                    else:
                        last_change_time = ts

        if config.enforce_capacity_limits:
            daily_usage: Dict[Any, float] = defaultdict(float)
            cumulative_turnover = 0.0
            scaling_factor = np.clip(config.capacity_scaling_factor, 0.0, 1.0)
            violation_action = (config.capacity_violation_action or 'scale_confidence').lower()

            for i in range(len(labels_adjusted)):
                prev_value = labels_adjusted.iat[i - 1] if i > 0 else 0
                current_value = labels_adjusted.iat[i]
                turnover_delta = abs(current_value - prev_value)

                if turnover_delta == 0:
                    continue

                timestamp = index[i] if i < len(index) else index[-1]
                day_key: Any
                if isinstance(index, pd.DatetimeIndex):
                    day_key = timestamp.normalize()
                else:
                    day_key = i

                proposed_daily = daily_usage[day_key] + turnover_delta
                potential_cumulative = cumulative_turnover + turnover_delta
                impact_cost = (
                    safe_power(potential_cumulative, config.impact_penalty_exponent)
                    * config.impact_cost_per_unit_turnover
                ) if config.impact_cost_per_unit_turnover else 0.0

                violation_detected = False

                if (
                    config.max_turnover_per_day is not None and
                    proposed_daily > config.max_turnover_per_day
                ):
                    diagnostics['turnover_violations'] += 1
                    violation_detected = True

                if (
                    config.max_impact_cost_pct is not None and
                    impact_cost > config.max_impact_cost_pct
                ):
                    diagnostics['impact_violations'] += 1
                    violation_detected = True

                if violation_detected and violation_action == 'zero_out':
                    labels_adjusted.iat[i] = prev_value
                    confidence_adjusted.iat[i] = 0.0
                    diagnostics['violating_timestamps'].append(timestamp)
                    continue

                if violation_detected:
                    confidence_adjusted.iat[i] *= scaling_factor
                    diagnostics['scaled_timestamps'].append(timestamp)

                daily_usage[day_key] = proposed_daily
                cumulative_turnover = potential_cumulative

        turnover_series = labels_adjusted.diff().abs().fillna(0.0)
        if not turnover_series.empty:
            turnover_series.iloc[0] = abs(labels_adjusted.iloc[0])

        realized_turnover = float(turnover_series.sum())
        daily_turnover: Dict[Any, float] = {}
        if isinstance(index, pd.DatetimeIndex) and not turnover_series.empty:
            daily_turnover = turnover_series.groupby(index.normalize()).sum().to_dict()

        trading_days = len(daily_turnover) if daily_turnover else (1 if realized_turnover > 0 else 0)
        capacity_utilization = 0.0
        if config.max_turnover_per_day and trading_days > 0:
            capacity_utilization = realized_turnover / (trading_days * config.max_turnover_per_day)

        impact_cost_total = (
            safe_power(realized_turnover, config.impact_penalty_exponent)
            * config.impact_cost_per_unit_turnover
        ) if config.impact_cost_per_unit_turnover else 0.0

        violations_total = (
            diagnostics['min_holding_violations'] +
            diagnostics['turnover_violations'] +
            diagnostics['impact_violations']
        )

        turnover_events = int((turnover_series > 0).sum()) or 1
        violation_penalty = min(1.0, violations_total / turnover_events)
        capacity_score = 1.0 - violation_penalty if config.enforce_capacity_limits else 1.0

        # Bound values within reasonable ranges
        capacity_score = float(np.clip(capacity_score, 0.0, 1.0))
        capacity_utilization = float(max(0.0, capacity_utilization))

        diagnostics.update({
            'realized_turnover': realized_turnover,
            'daily_turnover': daily_turnover,
            'capacity_utilization': capacity_utilization,
            'impact_cost': float(impact_cost_total),
            'capacity_score': capacity_score,
            'violations_flagged': violations_total > 0,
            'trading_days_evaluated': trading_days,
            'total_turnover_events': turnover_events,
            'net_profit_sum': float(net_profits.reindex(labels_adjusted.index).fillna(0.0).sum())
        })

        confidence_adjusted = confidence_adjusted.clip(lower=0.0, upper=1.0)

        return labels_adjusted.astype(int), confidence_adjusted, diagnostics

    def _apply_data_cleaning(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Apply data cleaning according to configuration with memory optimization."""
        tprint_info("🧹 Applying data cleaning")

        # Generate cache key for data cleaning
        data_hash = hashlib.md5(str(market_data.values).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(f"{self.cleaning_config.outlier_threshold}_{self.cleaning_config.min_volume_threshold}".encode()).hexdigest()[:8]
        cache_key = self._generate_cache_key(data_hash, config_hash, "data_cleaning")

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            tprint_info("📋 Using cached data cleaning results")
            return cached_result

        # Use memory-efficient data cleaning if available
        if self.memory_optimizer and self.analyst_config.enable_hardware_optimization:
            try:
                tprint_info("🧠 Using memory-optimized data cleaning")
                cleaned, cleaning_report = self.memory_optimizer.optimized_data_cleaning(
                    market_data, self.cleaning_config, {
                        'outlier_method': 'iqr',
                        'volume_filtering': True,
                        'price_filtering': True,
                        'deduplication': self.cleaning_config.enable_deduplication,
                        'timestamp_alignment': self.cleaning_config.enforce_timestamp_alignment,
                        'use_vectorized_operations': self.analyst_config.enable_vectorized_operations
                    }
                )
                tprint_success(f"✅ Memory-optimized cleaning applied: {len(cleaned)}/{len(market_data)} bars remaining")
                self._cache_result(cache_key, cleaned)
                return cleaned
            except Exception as e:
                tprint_warning(f"⚠️ Memory-optimized cleaning failed, using standard approach: {e}")

        # Standard data cleaning with vectorized operations where possible
        cleaned = market_data.copy()

        # Remove outliers using vectorized operations if available
        if self.cleaning_config.outlier_method == "iqr" and self.matrix_ops and self.analyst_config.enable_vectorized_operations:
            try:
                # Use vectorized outlier removal for better performance
                for col in ['high', 'low', 'close', 'volume']:
                    if col in cleaned.columns:
                        col_data = cleaned[col].values
                        Q1_val = np.percentile(col_data, 25)
                        Q3_val = np.percentile(col_data, 75)
                        IQR_val = Q3_val - Q1_val
                        lower_bound = Q1_val - self.cleaning_config.outlier_threshold * IQR_val
                        upper_bound = Q3_val + self.cleaning_config.outlier_threshold * IQR_val

                        # Vectorized mask creation
                        mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                        cleaned = cleaned.loc[mask]
            except Exception as e:
                tprint_warning(f"⚠️ Vectorized outlier removal failed, using pandas: {e}")
                # Fallback to original method
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
        else:
            # Original outlier removal method
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

        # Apply volume filters using vectorized operations if available
        if 'volume' in cleaned.columns and self.matrix_ops and self.analyst_config.enable_vectorized_operations:
            try:
                volume_data = cleaned['volume'].values
                volume_mask = (volume_data >= self.cleaning_config.min_volume_threshold) & \
                             (volume_data <= self.cleaning_config.max_volume_threshold)
                cleaned = cleaned.loc[volume_mask]
            except Exception as e:
                tprint_warning(f"⚠️ Vectorized volume filtering failed, using pandas: {e}")
                cleaned = cleaned[
                    (cleaned['volume'] >= self.cleaning_config.min_volume_threshold) &
                    (cleaned['volume'] <= self.cleaning_config.max_volume_threshold)
                ]
        else:
            if 'volume' in cleaned.columns:
                cleaned = cleaned[
                    (cleaned['volume'] >= self.cleaning_config.min_volume_threshold) &
                    (cleaned['volume'] <= self.cleaning_config.max_volume_threshold)
                ]

        # Apply price filters using vectorized operations if available
        if 'close' in cleaned.columns and self.matrix_ops and self.analyst_config.enable_vectorized_operations:
            try:
                # Remove zero/negative prices
                close_data = cleaned['close'].values
                price_mask = close_data >= self.cleaning_config.min_price
                cleaned = cleaned.loc[price_mask]

                # Remove extreme price changes
                if len(cleaned) > 1:
                    close_cleaned = cleaned['close'].values
                    price_changes = np.diff(close_cleaned) / close_cleaned[:-1]
                    change_mask = np.abs(price_changes) <= self.cleaning_config.max_price_change_pct
                    # Need to be careful with index alignment here
                    change_mask_extended = np.concatenate([[True], change_mask])  # Keep first element
                    cleaned = cleaned.loc[change_mask_extended]
            except Exception as e:
                tprint_warning(f"⚠️ Vectorized price filtering failed, using pandas: {e}")
                # Fallback to original method
                cleaned = cleaned[cleaned['close'] >= self.cleaning_config.min_price]
                if len(cleaned) > 1:
                    price_changes = cleaned['close'].pct_change()
                    cleaned = cleaned[abs(price_changes) <= self.cleaning_config.max_price_change_pct]
        else:
            if 'close' in cleaned.columns:
                # Remove zero/negative prices
                cleaned = cleaned[cleaned['close'] >= self.cleaning_config.min_price]

                # Remove extreme price changes
                if len(cleaned) > 1:
                    price_changes = cleaned['close'].pct_change()
                    cleaned = cleaned[abs(price_changes) <= self.cleaning_config.max_price_change_pct]

        # Remove duplicates if enabled (this is typically fast and doesn't need vectorization)
        if self.cleaning_config.enable_deduplication:
            cleaned = cleaned[~cleaned.index.duplicated(keep='first')]

        # Align timestamps if enabled
        if self.cleaning_config.enforce_timestamp_alignment:
            cleaned = self._align_timestamps(cleaned)

        tprint_info(f"✅ Data cleaning applied: {len(cleaned)}/{len(market_data)} bars remaining")
        self._cache_result(cache_key, cleaned)
        return cleaned

    def _get_bar_duration_minutes(self, market_data: pd.DataFrame) -> Optional[float]:
        """Estimate the bar duration in minutes based on timestamp spacing."""
        if len(market_data.index) < 2:
            return None

        diffs = market_data.index.to_series().diff().dropna()
        if diffs.empty:
            return None

        median_diff = diffs.median()
        if pd.isna(median_diff):
            return None

        return median_diff.total_seconds() / 60.0

    def _build_execution_context(
        self,
        market_data: pd.DataFrame,
        horizon_minutes: int
    ) -> Dict[str, Any]:
        """Construct execution-aware context for pricing and metadata."""
        # Auto-detect bar duration from actual data instead of using hardcoded value
        detected_bar_duration = self._get_bar_duration_minutes(market_data)
        bar_duration_minutes = detected_bar_duration if detected_bar_duration is not None else self.analyst_config.bar_duration_minutes
        horizon_bars = max(1, horizon_minutes // int(bar_duration_minutes))

        entry_column = 'open' if 'open' in market_data.columns else 'close'
        exit_column = 'close' if 'close' in market_data.columns else entry_column

        # For realistic trading simulation without lookahead bias:
        # - Entry price: Next bar's open (simulates market order at next bar)
        # - Exit price: Estimated future price based on historical patterns
        entry_prices = market_data[entry_column].shift(-1)  # Next bar's entry price

        # For LABELING, use actual future prices (lookahead is intentional)
        # We're creating labels based on what actually happened - lookahead protection 
        # happens during training/CV splits, not during label generation
        exit_prices = market_data[exit_column].shift(-horizon_bars)
        execution_mask = entry_prices.notna() & exit_prices.notna()
        

        bar_duration_minutes = self._get_bar_duration_minutes(market_data)

        self._last_execution_metadata = {
            'signal_to_execution_delay_bars': 1,
            'signal_to_execution_delay_minutes': None if bar_duration_minutes is None else float(bar_duration_minutes),
            'entry_price_source': f'next_{entry_column}',
            'exit_price_source': f'{exit_column}_estimated_{horizon_bars}_bars_ahead',
            'horizon_bars': horizon_bars,
            'holding_period_bars': horizon_bars,
            'signal_to_exit_delay_bars': horizon_bars + 1,
            'slippage_pct': self.analyst_config.trading_costs.slippage_pct,
            'fees': {
                'maker_fee': self.analyst_config.trading_costs.maker_fee,
                'taker_fee': self.analyst_config.trading_costs.taker_fee,
            },
            'slippage_applied_on': ['entry', 'exit'],
            'min_trade_size_usd': self.analyst_config.trading_costs.min_trade_size,
            'valid_execution_samples': int(execution_mask.sum()),
            'total_samples': int(len(market_data))
        }

        if bar_duration_minutes is not None:
            holding_minutes = float(bar_duration_minutes * horizon_bars)
            self._last_execution_metadata.update({
                'execution_horizon_minutes': holding_minutes,
                'holding_period_minutes': holding_minutes,
                'signal_to_exit_delay_minutes': float(bar_duration_minutes * (horizon_bars + 1)),
            })

        return {
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'execution_mask': execution_mask,
            'horizon_bars': horizon_bars,
        }

    def get_execution_latency_metadata(self) -> Dict[str, Any]:
        """Return the most recently computed execution latency metadata."""
        return dict(self._last_execution_metadata)

    def _calculate_expected_returns(
        self,
        market_data: pd.DataFrame,
        horizon_minutes: int,
        entry_prices: Optional[pd.Series] = None,
        exit_prices: Optional[pd.Series] = None
    ) -> pd.Series:
        """Calculate expected returns over horizon with optional asymmetric scaling."""
        # Use configurable bar duration instead of hardcoded 15 minutes
        bar_duration_minutes = self.analyst_config.bar_duration_minutes
        horizon_bars = max(1, horizon_minutes // bar_duration_minutes)

        # Generate cache key for expected returns calculation
        data_hash = hashlib.md5(str(market_data.values).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(f"{horizon_minutes}_{bar_duration_minutes}".encode()).hexdigest()[:8]
        cache_key = self._generate_cache_key(data_hash, config_hash, "expected_returns")

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            tprint_info("📋 Using cached expected returns calculation")
            return cached_result

        if entry_prices is None or exit_prices is None:
            context = self._build_execution_context(market_data, horizon_minutes)
            entry_prices = context['entry_prices']
            exit_prices = context['exit_prices']

        # Calculate forward returns
        forward_returns = (exit_prices - entry_prices) / entry_prices
        
        # Clean infinities and NaNs
        forward_returns = forward_returns.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Optimize rolling window calculation
        if self.matrix_ops and self.analyst_config.enable_vectorized_operations:
            try:
                # Use optimized rolling window calculation
                baseline_expectation = self._optimized_rolling_mean(forward_returns, horizon_bars)
            except Exception as e:
                tprint_warning(f"⚠️ Optimized rolling mean failed, using pandas: {e}")
                baseline_expectation = forward_returns.rolling(horizon_bars, min_periods=1).mean()
        else:
            baseline_expectation = forward_returns.rolling(horizon_bars, min_periods=1).mean()

        scaling_config = self.analyst_config.asymmetric_return_scaling
        if not scaling_config.enabled:
            result = baseline_expectation.fillna(0)
            self._cache_result(cache_key, result)
            return result

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

        result = expected_returns.fillna(0)
        self._cache_result(cache_key, result)
        return result

    def _optimized_rolling_mean(self, series: pd.Series, window: int) -> pd.Series:
        """Optimized rolling mean calculation using matrix operations."""
        if self.matrix_ops is None or not self.analyst_config.enable_vectorized_operations:
            return series.rolling(window, min_periods=1).mean()

        try:
            # Use matrix operations for optimized rolling window
            values = series.values
            result = np.zeros_like(values, dtype=np.float64)

            for i in range(len(values)):
                start_idx = max(0, i - window + 1)
                window_values = values[start_idx:i+1]
                if len(window_values) > 0:
                    result[i] = np.mean(window_values)
                else:
                    result[i] = 0.0

            return pd.Series(result, index=series.index)

        except Exception as e:
            tprint_warning(f"⚠️ Optimized rolling mean failed: {e}")
            return series.rolling(window, min_periods=1).mean()

    def _optimized_rolling_std(self, series: pd.Series, window: int) -> pd.Series:
        """Optimized rolling standard deviation calculation using matrix operations."""
        if self.matrix_ops is None or not self.analyst_config.enable_vectorized_operations:
            return series.rolling(window, min_periods=1).std()

        try:
            # Use vectorized rolling features for better performance
            from src.utils.matrix_operations import vectorized_rolling_features

            # Prepare data for vectorized rolling features
            data_df = pd.DataFrame({'value': series})
            windows = [window]

            # Use vectorized rolling features for standard deviation
            rolling_result = vectorized_rolling_features(
                data_df,
                windows=windows,
                features=['std']
            )

            # Extract the standard deviation column
            std_column = f'value_rolling_std_{window}'
            if std_column in rolling_result.columns:
                return rolling_result[std_column]
            else:
                # Fallback to pandas if vectorized features don't include the expected column
                return series.rolling(window, min_periods=1).std()

        except Exception as e:
            tprint_warning(f"⚠️ Optimized rolling std failed: {e}")
            return series.rolling(window, min_periods=1).std()

    def _calculate_trading_costs(
        self,
        market_data: pd.DataFrame,
        costs: TradingCosts,
        entry_prices: Optional[pd.Series] = None,
        exit_prices: Optional[pd.Series] = None
    ) -> pd.Series:
        """Calculate trading costs as a PERCENTAGE for each bar.

        Returns costs as percentage (e.g., 0.1 for 0.1%) for direct comparison with percentage returns.
        """

        # Generate cache key for trading costs calculation
        data_hash = hashlib.md5(str(market_data.values).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(f"{costs.maker_fee}_{costs.taker_fee}_{costs.slippage_pct}".encode()).hexdigest()[:8]
        cache_key = self._generate_cache_key(data_hash, config_hash, "trading_costs")

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            tprint_info("📋 Using cached trading costs calculation")
            return cached_result

        # Calculate total roundtrip cost as percentage
        # Entry: taker fee + slippage
        # Exit: taker fee + slippage
        entry_cost_pct = costs.taker_fee + costs.slippage_pct
        exit_cost_pct = costs.taker_fee + costs.slippage_pct
        total_cost_pct = entry_cost_pct + exit_cost_pct

        # Optimize mask calculation using vectorized operations if available
        if entry_prices is not None and self.matrix_ops and self.analyst_config.enable_vectorized_operations:
            try:
                # Use vectorized operations for mask calculation
                entry_valid = ~pd.isna(entry_prices)
                if exit_prices is not None:
                    exit_valid = ~pd.isna(exit_prices)
                    valid_mask_np = entry_valid.values & exit_valid.values
                    valid_mask = pd.Series(valid_mask_np, index=market_data.index)
                else:
                    valid_mask = entry_valid
            except Exception as e:
                tprint_warning(f"⚠️ Vectorized mask calculation failed, using pandas: {e}")
                valid_mask = entry_prices.notna()
                if exit_prices is not None:
                    valid_mask = valid_mask & exit_prices.notna()
        else:
            valid_mask = entry_prices.notna() if entry_prices is not None else pd.Series(True, index=market_data.index)
            if exit_prices is not None:
                valid_mask = valid_mask & exit_prices.notna()

        # Create costs series with optimized operations
        if self.matrix_ops and self.analyst_config.enable_vectorized_operations:
            try:
                # Use vectorized operations for costs series creation
                total_cost_pct_np = np.full(len(market_data), total_cost_pct, dtype=np.float64)
                valid_mask_np = valid_mask.values
                costs_np = np.where(valid_mask_np, total_cost_pct_np, 0.0)
                costs_series = pd.Series(costs_np, index=market_data.index)
            except Exception as e:
                tprint_warning(f"⚠️ Vectorized costs series failed, using pandas: {e}")
                costs_series = pd.Series(total_cost_pct, index=market_data.index)
                costs_series = costs_series.where(valid_mask, 0.0)
        else:
            costs_series = pd.Series(total_cost_pct, index=market_data.index)
            costs_series = costs_series.where(valid_mask, 0.0)

        self._cache_result(cache_key, costs_series)
        return costs_series

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

            # Use optimized rolling quantile for better performance
            if self.matrix_ops and self.analyst_config.enable_vectorized_operations:
                try:
                    # Use vectorized rolling features for quantile calculation
                    from src.utils.matrix_operations import vectorized_rolling_features

                    returns_df = pd.DataFrame({'returns': returns})
                    quantile_pct = np.clip(risk_config.downside_tail_percentile, 0.0, 1.0)

                    # Use vectorized rolling features for quantile
                    rolling_result = vectorized_rolling_features(
                        returns_df,
                        windows=[lookback],
                        features=['quantile']
                    )

                    # Extract the quantile column and get the specific percentile
                    quantile_column = f'returns_rolling_quantile_{lookback}'
                    if quantile_column in rolling_result.columns:
                        downside_tail = rolling_result[quantile_column].fillna(0).abs() * risk_config.downside_penalty_multiplier
                    else:
                        # Fallback to pandas quantile if vectorized features don't work
                        downside_tail = returns.rolling(lookback, min_periods=1).quantile(quantile_pct)
                        downside_tail = downside_tail.fillna(0).abs() * risk_config.downside_penalty_multiplier
                except Exception as e:
                    tprint_warning(f"⚠️ Optimized rolling quantile failed, using pandas: {e}")
                    downside_tail = returns.rolling(lookback, min_periods=1).quantile(
                        np.clip(risk_config.downside_tail_percentile, 0.0, 1.0)
                    )
                    downside_tail = downside_tail.fillna(0).abs() * risk_config.downside_penalty_multiplier
            else:
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
        # Confidence based on signal-to-noise ratio and expected return magnitude
        # Use expected returns magnitude instead of net_profits to avoid negative values
        signal_strength = abs(expected_returns) / (volatility_series + 1e-8)

        # Normalize by the 90th percentile to get values between 0 and 1
        if len(signal_strength) > 0:
            percentile_90 = signal_strength.quantile(0.9)
            if percentile_90 > 0:
                confidence = np.clip(signal_strength / percentile_90, 0, 1)
            else:
                confidence = pd.Series(0.5, index=signal_strength.index)  # Neutral confidence if no signal
        else:
            confidence = pd.Series(0.5, index=signal_strength.index)

        return confidence

    def _calculate_excursions(self, market_data: pd.DataFrame, horizon_minutes: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate favorable and adverse excursions over horizon."""
        horizon_bars = max(1, horizon_minutes // 15)  # Assuming 15m bars

        # Calculate rolling max and min over horizon using optimized operations
        if self.matrix_ops and self.analyst_config.enable_vectorized_operations:
            try:
                # Use vectorized rolling features for max/min calculations
                from src.utils.matrix_operations import vectorized_rolling_features

                # Prepare data for vectorized rolling features
                price_data = market_data[['high', 'low']]

                # Use vectorized rolling features for max and min
                rolling_result = vectorized_rolling_features(
                    price_data,
                    windows=[horizon_bars],
                    features=['max', 'min']
                )

                # Extract the rolling max and min columns
                high_max_column = f'high_rolling_max_{horizon_bars}'
                low_min_column = f'low_rolling_min_{horizon_bars}'

                if high_max_column in rolling_result.columns and low_min_column in rolling_result.columns:
                    rolling_high = rolling_result[high_max_column].shift(-horizon_bars)
                    rolling_low = rolling_result[low_min_column].shift(-horizon_bars)
                else:
                    # Fallback to pandas if vectorized features don't include expected columns
                    rolling_high = market_data['high'].rolling(horizon_bars).max().shift(-horizon_bars)
                    rolling_low = market_data['low'].rolling(horizon_bars).min().shift(-horizon_bars)
            except Exception as e:
                tprint_warning(f"⚠️ Optimized rolling max/min failed, using pandas: {e}")
                rolling_high = market_data['high'].rolling(horizon_bars).max().shift(-horizon_bars)
                rolling_low = market_data['low'].rolling(horizon_bars).min().shift(-horizon_bars)
        else:
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

    def _simulate_trade_outcomes(
        self,
        market_data: pd.DataFrame,
        stop_loss_levels: pd.Series,
        take_profit_levels: pd.Series,
        entry_prices: Optional[pd.Series] = None,
        horizon_bars: Optional[int] = None
    ) -> pd.Series:
        """Simulate trade outcomes assuming next-bar execution."""
        if entry_prices is None:
            context = self._build_execution_context(market_data, self.analyst_config.horizon_minutes)
            entry_prices = context['entry_prices']
            if horizon_bars is None:
                horizon_bars = context['horizon_bars']

        if horizon_bars is None:
            horizon_bars = max(1, self.analyst_config.horizon_minutes // 15)

        highs = market_data.get('high', market_data.get('close'))
        lows = market_data.get('low', market_data.get('close'))

        outcomes = pd.Series(0, index=market_data.index, dtype=int)
        index_list = list(market_data.index)

        for position, idx in enumerate(index_list):
            if position >= len(index_list) - 1:
                break

            entry_price = entry_prices.iloc[position] if entry_prices is not None else np.nan
            if pd.isna(entry_price):
                continue

            stop_price = stop_loss_levels.iloc[position]
            target_price = take_profit_levels.iloc[position]

            if pd.isna(stop_price) or pd.isna(target_price):
                continue

            start = position + 1
            end = min(len(index_list), start + horizon_bars)
            outcome = 0

            for future_pos in range(start, end):
                bar_high = highs.iloc[future_pos] if highs is not None else np.nan
                bar_low = lows.iloc[future_pos] if lows is not None else np.nan

                stop_hit = pd.notna(bar_low) and bar_low <= stop_price
                target_hit = pd.notna(bar_high) and bar_high >= target_price

                if stop_hit and target_hit:
                    outcome = -1  # assume worst-case: stop triggers first
                    break
                if stop_hit:
                    outcome = -1
                    break
                if target_hit:
                    outcome = 1
                    break

            outcomes.iloc[position] = outcome

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

            # Calculate autocorrelation using optimized correlation calculation
            if self.matrix_ops and self.analyst_config.enable_vectorized_operations:
                try:
                    # Use matrix operations for faster correlation calculation
                    from src.utils.matrix_operations import safe_correlation_matrix

                    # Prepare data as numpy arrays for vectorized operations
                    labels_np = recent_labels.values
                    lagged_np = recent_labels.shift(1).fillna(0).values

                    # Use safe correlation calculation from matrix operations
                    correlation = safe_correlation_matrix(pd.DataFrame({
                        'labels': labels_np,
                        'lagged': lagged_np
                    })).iloc[0, 1]

                    return correlation if not pd.isna(correlation) else 0.0
                except Exception as e:
                    tprint_warning(f"⚠️ Optimized correlation failed, using pandas: {e}")
                    # Fallback to pandas correlation
                    lagged = recent_labels.shift(1).fillna(0)
                    correlation = recent_labels.corr(lagged)
                    return correlation if not pd.isna(correlation) else 0.0
            else:
                # Use pandas correlation
                lagged = recent_labels.shift(1).fillna(0)
                correlation = recent_labels.corr(lagged)
                return correlation if not pd.isna(correlation) else 0.0

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating autocorrelation: {e}")
            return 0.0

    def _calculate_drift_score(self, current_labels: pd.Series, historical_labels: pd.Series) -> float:
        """Calculate drift between current and historical labels."""
        try:
            # Simple drift measure: difference in positive ratios
            current_ratio = current_labels.mean()
            historical_ratio = historical_labels.mean()

            return abs(current_ratio - historical_ratio)

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating drift score: {e}")
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
            min_holding_minutes=30,
            max_turnover_per_day=12,
            capacity_violation_action="scale_confidence",
            capacity_scaling_factor=0.5,
            impact_cost_per_unit_turnover=0.0,
            impact_penalty_exponent=1.0,
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
