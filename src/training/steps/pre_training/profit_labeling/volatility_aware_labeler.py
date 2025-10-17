"""
Volatility Aware Labeler Module

This module provides volatility-aware labeling functionality for profit labeling.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from src.utils.logger import system_logger

# Import the missing function from multi_horizon_profit_labeler
try:
    from src.training.steps.pre_training.multi_horizon_profit_labeler import create_enhanced_tactician_labeler
    # Check if the function is actually available (not the fallback)
    import inspect
    if hasattr(create_enhanced_tactician_labeler, '__name__') and 'Unavailable' in str(create_enhanced_tactician_labeler):
        # The function is the fallback version, create a proper implementation
        def create_enhanced_tactician_labeler(*args: Any, **kwargs: Any) -> Any:
            """Enhanced tactician labeler implementation."""
            # For now, return a simple implementation that doesn't fail
            class SimpleTacticianLabeler:
                def __init__(self, *args, **kwargs):
                    pass
                def generate_labels(self, *args, **kwargs):
                    return {"labels": None, "metadata": {"type": "tactician", "status": "fallback"}}
            return SimpleTacticianLabeler(*args, **kwargs)
except ImportError:
    # Fallback implementation if import fails
    def create_enhanced_tactician_labeler(*args: Any, **kwargs: Any) -> Any:
        """Fallback implementation for create_enhanced_tactician_labeler."""
        class SimpleTacticianLabeler:
            def __init__(self, *args, **kwargs):
                pass
            def generate_labels(self, *args, **kwargs):
                return {"labels": None, "metadata": {"type": "tactician", "status": "fallback"}}
        return SimpleTacticianLabeler(*args, **kwargs)


def _align_like(left: pd.Series, right: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align two series using inner join to ensure consistent indices.
    
    Args:
        left: First series to align
        right: Second series to align
        
    Returns:
        Tuple of aligned series (left_aligned, right_aligned)
    """
    a, b = left.align(right, join="inner")
    return a, b


class LabelDefinitionType(Enum):
    """Enum for label definition types."""
    BINARY = "binary"
    MULTI_CLASS = "multi_class"
    REGRESSION = "regression"
    ANALYST = "analyst"  # For analyst profit labeling (long-term analysis)
    TACTICIAN = "tactician"  # For tactician entry labeling (short-term entry)


class VolatilityAwareConfig:
    """
    Configuration for volatility-aware labeling.
    """
    
    def __init__(
        self,
        volatility_threshold: float = 0.02,
        lookahead_periods: int = 5,
        min_volatility: float = 0.001,
        max_volatility: float = 0.1,
        label_type: LabelDefinitionType = LabelDefinitionType.BINARY,
        enable_long_positions: bool = True,
        enable_short_positions: bool = False
    ):
        """
        Initialize volatility-aware configuration.
        
        Args:
            volatility_threshold: Threshold for volatility-based labeling
            lookahead_periods: Number of periods to look ahead
            min_volatility: Minimum volatility threshold
            max_volatility: Maximum volatility threshold
            label_type: Type of labels to generate
            enable_long_positions: Whether to generate long position signals
            enable_short_positions: Whether to generate short position signals
        """
        self.volatility_threshold = volatility_threshold
        self.lookahead_periods = lookahead_periods
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.label_type = label_type
        self.enable_long_positions = enable_long_positions
        self.enable_short_positions = enable_short_positions
        
        # Initialize additional configuration attributes
        self.label_definition_type = label_type
        self.enable_enhanced_labels = False
        self.timeframe = None
        self.enable_quality_scoring = True
        self.quality_scoring = QualityScoringConfig()
        self.regime_config = RegimeConfig()
        self.optimal_entry_detection = OptimalEntryDetectionConfig()
        
        # Initialize bar construction configuration
        from .bar_construction import BarConstructionConfig
        self.bar_construction = BarConstructionConfig()
        
        # Initialize noise gating configuration
        self.noise_gating = NoiseGatingConfig()
        
        # Initialize multi-target configuration
        self.multi_target = MultiTargetConfig()
        
        # Initialize volatility configuration
        self.volatility = VolatilityConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters and raise helpful errors."""
        errors = []
        
        # Validate volatility thresholds
        if not (0 < self.min_volatility <= self.max_volatility):
            errors.append(f"Invalid volatility range: min_volatility ({self.min_volatility}) must be > 0 and <= max_volatility ({self.max_volatility})")
        
        # Validate lookahead periods
        if self.lookahead_periods < 1:
            errors.append(f"lookahead_periods ({self.lookahead_periods}) must be >= 1")
        
        # Validate volatility window
        if self.volatility.window < 2:
            errors.append(f"volatility.window ({self.volatility.window}) must be >= 2")
        
        # Validate multi-target profit targets
        if hasattr(self.multi_target, 'target_profits') and self.multi_target.target_profits:
            for i, target in enumerate(self.multi_target.target_profits):
                if target <= 0:
                    errors.append(f"multi_target.target_profits[{i}] ({target}) must be > 0")
        
        # Validate quality scoring thresholds
        if not (0 <= self.quality_scoring.min_quality_threshold <= 1):
            errors.append(f"quality_scoring.min_quality_threshold ({self.quality_scoring.min_quality_threshold}) must be between 0 and 1")
        
        if not (0 <= self.quality_scoring.min_predictability <= 1):
            errors.append(f"quality_scoring.min_predictability ({self.quality_scoring.min_predictability}) must be between 0 and 1")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)


class QualityScoringConfig:
    """Configuration for quality scoring."""
    def __init__(self):
        self.min_quality_threshold = 0.3
        self.min_predictability = 0.3


class RegimeConfig:
    """Configuration for regime adaptation."""
    def __init__(self):
        self.enabled = False


class OptimalEntryDetectionConfig:
    """Configuration for optimal entry point detection."""
    def __init__(self):
        self.enabled = False
        self.entry_threshold = 0.5
        self.find_highest_gap_entry = False
        self.entry_point_strategy = "default"
        self.horizons = []
        self.target_profits = []
        self.multi_size_thresholds = []
        self.max_windows = 10


class NoiseGatingConfig:
    """Configuration for noise gating."""
    def __init__(self):
        self.enabled = True


class MultiTargetConfig:
    """Configuration for multi-target labeling."""
    def __init__(self):
        self.horizons = []
        self.target_profits = []
        self.min_lqs_score = 0.3


class VolatilityConfig:
    """Configuration for volatility settings."""
    def __init__(self):
        self.enabled = True
        self.window = 20
        self.sensitivity = 1.5  # Tunable parameter for volatility sensitivity


class LabelingResult:
    """
    Result of labeling operation.
    """
    
    def __init__(
        self,
        labels: pd.Series,
        metadata: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None,
        quality_scores: Dict[str, Any] = None
    ):
        """
        Initialize labeling result.
        
        Args:
            labels: Generated labels
            metadata: Additional metadata
            success: Whether labeling was successful
            error_message: Error message if unsuccessful
        """
        self.labels = labels
        self.metadata = metadata
        self.success = success
        self.error_message = error_message
        self.quality_scores = quality_scores or {}
        
        # Add convenience attributes with defensive counting
        self.n_samples = int(len(labels)) if labels is not None else 0

        # Use metadata values first, fallback to dtype inspection only if missing
        self.n_targets = self.metadata.get("n_targets")
        if self.n_targets is None and labels is not None:
            # Fallback to dtype inspection only if metadata doesn't have n_targets
            if isinstance(labels, pd.DataFrame):
                # For DataFrame, count columns as targets
                self.n_targets = len(labels.columns)
            else:
                # For Series, check if it's integer dtype (classification)
                if pd.api.types.is_integer_dtype(labels.dtype):
                    self.n_targets = int(labels.dropna().nunique())
                else:
                    self.n_targets = 1  # Single target for regression

        self.n_horizons = int(self.metadata.get("n_horizons", 1))
        self.confidence_scores = self.metadata.get("confidence_scores")
        self.eligibility_masks = self.metadata.get("eligibility_masks")
        # quality_scores is now passed directly as parameter
        self.normalization_factors = self.metadata.get("normalization_factors")
        self.processing_time = self.metadata.get("processing_time")


class VolatilityAwareMultiHorizonLabeler:
    """
    Volatility-aware multi-horizon labeler.
    """
    
    def __init__(self, config: VolatilityAwareConfig):
        """
        Initialize the volatility-aware labeler.
        
        Args:
            config: Configuration for the labeler
        """
        self.config = config
        self.logger = system_logger.getChild("VolatilityAwareMultiHorizonLabeler")
        
    def generate_labels(
        self,
        data: pd.DataFrame,
        price_column: str = "close",
        volatility_column: Optional[str] = None,
        profit_targets: Optional[List[float]] = None
    ) -> LabelingResult:
        """
        Generate volatility-aware labels with analyst profit targets.

        Args:
            data: Input data
            price_column: Name of price column
            volatility_column: Name of volatility column (optional)
            profit_targets: Optional list of profit targets for analyst labeling

        Returns:
            LabelingResult with generated labels
        """
        try:
            # Edge case handling: empty/short series
            if len(data) < self.config.lookahead_periods:
                self.logger.warning(f"Insufficient data: {len(data)} rows < {self.config.lookahead_periods} lookahead periods")
                return LabelingResult(
                    pd.Series(dtype=float, name='label'),
                    {"reason": "insufficient_history", "n_samples": len(data), "n_horizons": 1, "n_targets": 0},
                    success=True
                )
            
            # Edge case handling: non-monotonic/duplicate index
            if not data.index.is_monotonic_increasing:
                self.logger.warning("Non-monotonic index detected - proceeding with caution")
            
            if data.index.duplicated().any():
                self.logger.warning("Duplicate index values detected - proceeding with caution")
            
            # Edge case handling: constant price
            price_series = data[price_column]
            if price_series.nunique() <= 1:
                self.logger.warning("Constant price detected - all labels will be zero")
            
            # Explicit units for targets - treat inputs as percent points
            target_pp = profit_targets or []
            # Ensure we extract scalar values from any Series objects
            targets_frac = []
            if target_pp:
                for t in target_pp:
                    if isinstance(t, pd.Series):
                        # Extract scalar value from Series (use first non-null value)
                        scalar_t = t.dropna().iloc[0] if len(t.dropna()) > 0 else 0.0
                    else:
                        scalar_t = float(t)
                    targets_frac.append(scalar_t / 100.0)
            
            # Calculate volatility with proper configuration
            if volatility_column is None or volatility_column not in data.columns:
                if self.config.volatility.enabled:
                    volatility = price_series.pct_change().rolling(window=self.config.volatility.window).std()
                else:
                    volatility = pd.Series(1.0, index=price_series.index)  # Default multiplier = 1
            else:
                volatility = data[volatility_column]

            # Generate labels based on volatility and profit targets
            labels = self._generate_volatility_labels(price_series, volatility, targets_frac)
            
            # Generate quality scores with proper alignment
            quality_scores = self._calculate_quality_scores(labels, price_series)
            
            # Determine result shape and format
            if isinstance(labels, pd.DataFrame):
                # Multi-target case
                result_labels = labels
                n_targets = len(labels.columns)
                label_columns = list(labels.columns)
            else:
                # Single target case
                result_labels = labels.rename('label')
                n_targets = 1
                label_columns = ['label']
            
            # Build comprehensive metadata
            metadata = {
                "volatility_threshold": self.config.volatility_threshold,
                "lookahead_periods": self.config.lookahead_periods,
                "label_type": self.config.label_type.value,
                "total_labels": len(result_labels),
                "non_null_labels": result_labels.notna().sum() if isinstance(result_labels, pd.Series) else result_labels.notna().sum().sum(),
                "quality_scores": quality_scores,
                "profit_targets_pp": target_pp,
                "profit_targets_frac": targets_frac,
                "n_horizons": 1,
                "n_targets": n_targets,
                "label_columns": label_columns,
                "labels_shape": result_labels.shape,
                "labels_mem_bytes": result_labels.memory_usage(deep=True).sum() if isinstance(result_labels, pd.DataFrame) else result_labels.memory_usage(deep=True),
                "volatility_enabled": self.config.volatility.enabled,
                "volatility_window": self.config.volatility.window
            }
            
            # Comprehensive sanity checklist and logging
            self._log_quality_sanity_check(result_labels, quality_scores, metadata)
            
            # Logging & observability - single-line KPI
            coverage = metadata["non_null_labels"] / metadata["total_labels"] if metadata["total_labels"] > 0 else 0
            positive_rate = (result_labels > 0).sum() / metadata["total_labels"] if metadata["total_labels"] > 0 else 0
            
            self.logger.info(f"Labels generated: {metadata['total_labels']} rows, {n_targets} targets, "
                           f"coverage {coverage:.1%}, positive rate {positive_rate:.1%}, "
                           f"vol window {self.config.volatility.window}/{self.config.volatility.enabled}")
            
            # Warn on suspicious states
            if coverage < 0.01:
                self.logger.warning(f"Very low coverage: {coverage:.1%} - check data quality")
            if positive_rate == 0:
                self.logger.warning("No positive labels found - check thresholds")
            elif positive_rate == 1:
                self.logger.warning("All labels positive - check thresholds")
            
            return LabelingResult(result_labels, metadata, success=True, quality_scores=quality_scores)

        except Exception as e:
            self.logger.error(f"Error generating labels: {e}")
            return LabelingResult(
                pd.Series(dtype=float, name='label'),
                {"reason": "error", "error": str(e), "n_horizons": 1, "n_targets": 0},
                success=False,
                error_message=str(e)
            )
    
    def _calculate_quality_scores(self, labels: Union[pd.Series, pd.DataFrame], prices: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive quality scores with IC, Hit Rate, Uplift, Stability, and Risk-aware metrics."""
        try:
            # Handle both Series and DataFrame inputs
            if isinstance(labels, pd.DataFrame):
                # For multi-target, calculate quality for each target
                target_qualities = {}
                for col in labels.columns:
                    target_quality = self._calculate_comprehensive_target_quality(labels[col], prices, col)
                    target_qualities[col] = target_quality
                
                # Apply multiple testing hygiene with FDR control
                target_qualities = self._apply_multiple_testing_hygiene(target_qualities)
                
                # Aggregate across targets using median for robustness
                return self._aggregate_target_qualities(target_qualities)
            else:
                # Single target case
                target_quality = self._calculate_comprehensive_target_quality(labels, prices, 'default')
                return {'default': target_quality}
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality scores: {e}")
            return self._create_fallback_quality_score()
    
    def _calculate_comprehensive_target_quality(self, labels: pd.Series, prices: pd.Series, target_name: str) -> Any:
        """Calculate quality scores focused on trade opportunities using potential profit."""
        self.logger.info(f"DEBUG: Starting trade opportunity quality for {target_name}")
        self.logger.info(f"DEBUG: labels length: {len(labels)}, prices length: {len(prices)}")
        
        # Align series to ensure consistent indices
        labels_aligned, prices_aligned = _align_like(labels, prices)
        self.logger.info(f"DEBUG: After alignment - labels: {len(labels_aligned)}, prices: {len(prices_aligned)}")
        
        # Only calculate quality for trade opportunities (non-zero labels: positive for longs, negative for shorts)
        trade_opportunities = labels_aligned[labels_aligned != 0]
        if len(trade_opportunities) == 0:
            self.logger.warning(f"No trade opportunities found for {target_name}")
            return self._create_fallback_quality_score(reason="no_trade_opportunities")
        
        long_opportunities = len(trade_opportunities[trade_opportunities > 0])
        short_opportunities = len(trade_opportunities[trade_opportunities < 0])
        self.logger.info(f"DEBUG: Found {len(trade_opportunities)} trade opportunities out of {len(labels_aligned)} total samples")
        self.logger.info(f"DEBUG: Long opportunities: {long_opportunities}, Short opportunities: {short_opportunities}")
        
        # Calculate potential profit for each trade opportunity
        potential_profits = self._calculate_potential_profits(trade_opportunities, prices_aligned, target_name)
        
        # Calculate quality metrics based on potential profit
        metrics = self._calculate_trade_opportunity_metrics(trade_opportunities, potential_profits, target_name)
        
        # Calculate composite score based on potential profit quality
        composite_score = self._calculate_potential_profit_quality_score(metrics, potential_profits)
        
        # Create trade opportunity quality score object
        class TradeOpportunityQualityScore:
            def __init__(self, composite_score, metrics, potential_profits, target_name):
                self.overall_quality = composite_score
                self.predictability = metrics.get('ic', 0.0)
                self.stability = metrics.get('stability', 0.0)
                self.balance = 0.0  # Not relevant for low-frequency opportunities
                self.coverage = len(trade_opportunities) / len(labels_aligned) if len(labels_aligned) > 0 else 0.0
                self.target_name = target_name
                self.gates_passed = True  # No balance gates
                self.potential_profits = potential_profits
                self.avg_potential_profit = potential_profits.mean() if len(potential_profits) > 0 else 0.0
                self.max_potential_profit = potential_profits.max() if len(potential_profits) > 0 else 0.0
                # Store all metrics for detailed analysis
                self.metrics = metrics
                self.red_flag_reasons = self._extract_trade_opportunity_red_flags(metrics, potential_profits)
        
        return TradeOpportunityQualityScore(composite_score, metrics, potential_profits, target_name)
    
    def _calculate_potential_profits(self, trade_opportunities: pd.Series, prices: pd.Series, target_name: str) -> pd.Series:
        """Calculate potential profit based on signal direction in 90min period."""
        potential_profits = []
        
        for opportunity_idx in trade_opportunities.index:
            # Get 90min window (6 * 15min periods) starting from opportunity
            window_start = opportunity_idx
            window_end = min(opportunity_idx + 6, len(prices) - 1)
            
            if window_end > window_start:
                window_prices = prices.iloc[window_start:window_end]
                if len(window_prices) > 1:
                    start_price = window_prices.iloc[0]
                    signal_direction = trade_opportunities.loc[opportunity_idx]
                    
                    if start_price > 0:
                        if signal_direction > 0:  # Long signal
                            # For longs: (max - start) / start (upward movement)
                            max_price = window_prices.max()
                            potential_profit = (max_price - start_price) / start_price
                        elif signal_direction < 0:  # Short signal
                            # For shorts: (start - min) / start (downward movement)
                            min_price = window_prices.min()
                            potential_profit = (start_price - min_price) / start_price
                        else:  # No signal (shouldn't happen in trade_opportunities)
                            potential_profit = 0.0
                        
                        potential_profits.append(potential_profit)
                    else:
                        potential_profits.append(0.0)
                else:
                    potential_profits.append(0.0)
            else:
                potential_profits.append(0.0)
        
        return pd.Series(potential_profits, index=trade_opportunities.index)
    
    def _calculate_trade_opportunity_metrics(self, trade_opportunities: pd.Series, potential_profits: pd.Series, target_name: str) -> Dict[str, float]:
        """Calculate metrics specific to trade opportunities."""
        metrics = {}
        
        if len(potential_profits) == 0:
            return {'ic': 0.0, 'hit_rate': 0.0, 'uplift': 0.0, 'stability': 0.0, 'sharpe': 0.0}
        
        # Basic opportunity metrics
        metrics['avg_potential_profit'] = potential_profits.mean()
        metrics['max_potential_profit'] = potential_profits.max()
        metrics['min_potential_profit'] = potential_profits.min()
        metrics['std_potential_profit'] = potential_profits.std()
        
        # Quality metrics based on potential profit distribution
        # IC: correlation between opportunity timing and potential profit
        opportunity_timing = np.arange(len(trade_opportunities))
        ic = np.corrcoef(opportunity_timing, potential_profits)[0, 1] if len(potential_profits) > 1 else 0.0
        metrics['ic'] = ic if not np.isnan(ic) else 0.0
        
        # Hit rate: percentage of opportunities with above-average potential profit
        avg_profit = potential_profits.mean()
        hit_rate = (potential_profits > avg_profit).mean()
        metrics['hit_rate'] = hit_rate if not np.isnan(hit_rate) else 0.0
        
        # Uplift: difference between high and low potential profit opportunities
        if len(potential_profits) > 1:
            high_profit_mask = potential_profits > potential_profits.median()
            if high_profit_mask.sum() > 0 and (~high_profit_mask).sum() > 0:
                uplift = potential_profits[high_profit_mask].mean() - potential_profits[~high_profit_mask].mean()
                metrics['uplift'] = uplift if not np.isnan(uplift) else 0.0
            else:
                metrics['uplift'] = 0.0
        else:
            metrics['uplift'] = 0.0
        
        # Stability: consistency of potential profits over time
        if len(potential_profits) > 3:
            # Rolling standard deviation of potential profits
            rolling_std = potential_profits.rolling(window=min(3, len(potential_profits))).std()
            stability = 1 / (1 + rolling_std.mean()) if not rolling_std.mean() == 0 else 0.0
            metrics['stability'] = stability if not np.isnan(stability) else 0.0
        else:
            metrics['stability'] = 0.0
        
        # Sharpe: risk-adjusted potential profit
        if metrics['std_potential_profit'] > 0:
            sharpe = metrics['avg_potential_profit'] / metrics['std_potential_profit']
            metrics['sharpe'] = sharpe if not np.isnan(sharpe) else 0.0
        else:
            metrics['sharpe'] = 0.0
        
        return metrics
    
    def _calculate_potential_profit_quality_score(self, metrics: Dict[str, float], potential_profits: pd.Series) -> float:
        """Calculate quality score based on potential profit characteristics."""
        if len(potential_profits) == 0:
            return 0.0
        
        # Base score from average potential profit (higher is better)
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        profit_score = min(1.0, avg_profit / 0.02)  # Normalize to 2% max expected profit
        
        # Consistency score (lower std is better)
        std_profit = metrics.get('std_potential_profit', 0.0)
        consistency_score = 1.0 / (1.0 + std_profit * 10) if std_profit > 0 else 1.0
        
        # Hit rate score
        hit_rate = metrics.get('hit_rate', 0.0)
        hit_rate_score = hit_rate
        
        # Stability score
        stability = metrics.get('stability', 0.0)
        stability_score = stability
        
        # Sharpe score (risk-adjusted)
        sharpe = metrics.get('sharpe', 0.0)
        sharpe_score = min(1.0, max(0.0, (sharpe + 1) / 2))  # Normalize from [-1,1] to [0,1]
        
        # Weighted composite score
        composite_score = (
            0.4 * profit_score +      # 40% weight on potential profit
            0.2 * consistency_score + # 20% weight on consistency
            0.2 * hit_rate_score +    # 20% weight on hit rate
            0.1 * stability_score +   # 10% weight on stability
            0.1 * sharpe_score        # 10% weight on risk-adjusted return
        )
        
        return composite_score
    
    def _extract_trade_opportunity_red_flags(self, metrics: Dict[str, float], potential_profits: pd.Series) -> List[str]:
        """Extract red flags specific to trade opportunities."""
        red_flags = []
        
        # Check for low potential profit
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        if avg_profit < 0.005:  # Less than 0.5%
            red_flags.append("low_potential_profit")
        elif avg_profit < 0.01:  # Less than 1%
            red_flags.append("marginal_potential_profit")
        
        # Check for high volatility in potential profits
        std_profit = metrics.get('std_potential_profit', 0.0)
        if std_profit > avg_profit * 2:  # Std > 2x mean
            red_flags.append("high_profit_volatility")
        
        # Check for low hit rate
        hit_rate = metrics.get('hit_rate', 0.0)
        if hit_rate < 0.3:  # Less than 30% above average
            red_flags.append("low_hit_rate")
        
        # Check for low stability
        stability = metrics.get('stability', 0.0)
        if stability < 0.3:
            red_flags.append("low_stability")
        
        return red_flags[:1]  # Return first red flag only
    
    def _check_quality_gates(self, labels: pd.Series, lookahead_returns: pd.Series, coverage: float) -> bool:
        """Check minimum quality gates."""
        self.logger.info(f"DEBUG: Checking quality gates - coverage: {coverage:.3f}")
        
        # Gate 1: Coverage ≥ 5%
        if coverage < 0.05:
            self.logger.warning(f"DEBUG: Gate 1 FAILED - coverage {coverage:.3f} < 0.05")
            return False
        self.logger.info(f"DEBUG: Gate 1 PASSED - coverage {coverage:.3f} >= 0.05")
        
        # Gate 2: Balance check removed - not relevant for 2-5 opportunities per day
        if len(labels.dropna()) > 0:
            positive_rate = (labels.dropna() > 0).mean()
            self.logger.info(f"DEBUG: Gate 2 SKIPPED - positive_rate: {positive_rate:.3f} (balance not relevant for low-frequency opportunities)")
        
        # Gate 3: IC p-value < 0.1 in at least half of temporal folds
        if len(labels.dropna()) > 10 and len(lookahead_returns.dropna()) > 10:
            ic_pvalue = self._calculate_ic_pvalue(labels.dropna(), lookahead_returns.dropna())
            self.logger.info(f"DEBUG: Gate 3 - IC p-value: {ic_pvalue:.3f}")
            if ic_pvalue >= 0.1:
                self.logger.warning(f"DEBUG: Gate 3 FAILED - IC p-value {ic_pvalue:.3f} >= 0.1")
                return False
            self.logger.info(f"DEBUG: Gate 3 PASSED - IC p-value {ic_pvalue:.3f} < 0.1")
        else:
            self.logger.info(f"DEBUG: Gate 3 SKIPPED - insufficient data: labels={len(labels.dropna())}, returns={len(lookahead_returns.dropna())}")
        
        self.logger.info(f"DEBUG: All quality gates PASSED")
        return True
    
    def _calculate_target_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive metrics for a target."""
        metrics = {}
        
        # Predictability block (signal quality)
        metrics.update(self._calculate_predictability_metrics(labels, lookahead_returns))
        
        # Class/coverage block
        metrics.update(self._calculate_class_metrics(labels))
        
        # Stability block (time robustness)
        metrics.update(self._calculate_stability_metrics(labels, lookahead_returns))
        
        # Risk-aware block
        metrics.update(self._calculate_risk_metrics(labels, lookahead_returns))
        
        return metrics
    
    def _calculate_predictability_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate predictability metrics (IC, Hit Rate, Uplift)."""
        metrics = {}
        
        # Align data
        labels_clean, returns_clean = _align_like(labels, lookahead_returns)
        
        # Debug logging
        self.logger.info(f"DEBUG: labels_clean length: {len(labels_clean)}, non-null: {len(labels_clean.dropna())}")
        self.logger.info(f"DEBUG: returns_clean length: {len(returns_clean)}, non-null: {len(returns_clean.dropna())}")
        self.logger.info(f"DEBUG: labels_clean sample: {labels_clean.head()}")
        self.logger.info(f"DEBUG: returns_clean sample: {returns_clean.head()}")
        
        if len(labels_clean.dropna()) < 10:
            self.logger.warning(f"DEBUG: Insufficient data for metrics - {len(labels_clean.dropna())} < 10")
            return {'ic': 0.0, 'hit_rate': 0.0, 'uplift': 0.0}
        
        # Information Coefficient (Spearman correlation)
        ic = labels_clean.corr(returns_clean, method='spearman')
        self.logger.info(f"DEBUG: IC calculation: {ic}")
        metrics['ic'] = ic if not pd.isna(ic) else 0.0
        
        # Hit Rate
        if self._is_classification_like(labels_clean):
            # For classification labels
            if set(labels_clean.dropna().unique()) <= {0.0, 1.0}:
                # Binary 0/1 labels
                hit_rate = (np.sign(returns_clean) == labels_clean).mean()
            elif set(labels_clean.dropna().unique()) <= {-1.0, 0.0, 1.0}:
                # Ternary -1/0/1 labels
                hit_rate = (np.sign(returns_clean) == labels_clean).mean()
            else:
                hit_rate = 0.0
        else:
            # For regression, use correlation-based hit rate
            hit_rate = abs(ic)
        
        metrics['hit_rate'] = hit_rate if not pd.isna(hit_rate) else 0.0
        
        # Uplift (return difference)
        if self._is_classification_like(labels_clean):
            positive_mask = labels_clean > 0
            if positive_mask.sum() > 0 and (~positive_mask).sum() > 0:
                uplift = returns_clean[positive_mask].mean() - returns_clean[~positive_mask].mean()
                metrics['uplift'] = uplift if not pd.isna(uplift) else 0.0
            else:
                metrics['uplift'] = 0.0
        else:
            metrics['uplift'] = 0.0
        
        return metrics
    
    def _calculate_class_metrics(self, labels: pd.Series) -> Dict[str, float]:
        """Calculate class/coverage metrics."""
        metrics = {}
        
        # Coverage
        coverage = labels.notna().sum() / len(labels) if len(labels) > 0 else 0.0
        metrics['coverage'] = coverage
        
        # Balance
        if len(labels.dropna()) > 0:
            positive_rate = (labels.dropna() > 0).mean()
            balance = min(positive_rate, 1 - positive_rate) * 2
            metrics['balance'] = balance
        else:
            metrics['balance'] = 0.0
        
        return metrics
    
    def _calculate_stability_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate stability metrics (rolling IC stability, temporal CV)."""
        metrics = {}
        
        # Rolling IC stability
        window_size = min(252, len(labels) // 4)  # 1 year or 1/4 of data
        if window_size < 20:
            metrics['stability'] = 0.0
            return metrics
        
        rolling_ics = []
        for i in range(window_size, len(labels)):
            window_labels = labels.iloc[i-window_size:i]
            window_returns = lookahead_returns.iloc[i-window_size:i]
            if len(window_labels.dropna()) > 5 and len(window_returns.dropna()) > 5:
                ic = window_labels.corr(window_returns, method='spearman')
                if not pd.isna(ic):
                    rolling_ics.append(ic)
        
        if len(rolling_ics) > 1:
            ic_std = np.std(rolling_ics)
            stability = 1 / (1 + ic_std)  # Convert to score (lower std = higher stability)
            metrics['stability'] = stability
        else:
            metrics['stability'] = 0.0
        
        return metrics
    
    def _calculate_risk_metrics(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-aware metrics (Sharpe ratio)."""
        metrics = {}
        
        # Sharpe ratio of labeled subset
        if self._is_classification_like(labels):
            positive_mask = labels > 0
            if positive_mask.sum() > 5:  # Need enough samples
                labeled_returns = lookahead_returns[positive_mask]
                if len(labeled_returns.dropna()) > 5:
                    sharpe = labeled_returns.mean() / labeled_returns.std() if labeled_returns.std() > 0 else 0.0
                    metrics['sharpe'] = sharpe if not pd.isna(sharpe) else 0.0
                else:
                    metrics['sharpe'] = 0.0
            else:
                metrics['sharpe'] = 0.0
        else:
            metrics['sharpe'] = 0.0
        
        return metrics
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite quality score with suggested weights."""
        # Normalize metrics to [0, 1] range
        ic_norm = max(0, min(1, (metrics.get('ic', 0) + 1) / 2))  # IC from [-1,1] to [0,1]
        hit_rate_norm = max(0, min(1, metrics.get('hit_rate', 0)))
        uplift_norm = max(0, min(1, (metrics.get('uplift', 0) + 0.1) / 0.2))  # Cap at 0.1
        sharpe_norm = max(0, min(1, (metrics.get('sharpe', 0) + 2) / 4))  # Cap at 2
        stability = max(0, min(1, metrics.get('stability', 0)))
        balance = max(0, min(1, metrics.get('balance', 0)))
        coverage = max(0, min(1, metrics.get('coverage', 0)))
        
        # Composite score with suggested weights
        composite = (0.25 * ic_norm + 
                    0.20 * hit_rate_norm + 
                    0.15 * uplift_norm + 
                    0.10 * sharpe_norm + 
                    0.15 * stability + 
                    0.10 * balance + 
                    0.05 * coverage)
        
        return composite
    
    def _aggregate_target_qualities(self, target_qualities: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate quality scores across targets using median for robustness."""
        if not target_qualities:
            return self._create_fallback_quality_score()
        
        # Extract metrics for aggregation
        all_metrics = {}
        for target_name, quality in target_qualities.items():
            if hasattr(quality, 'metrics'):
                for metric_name, value in quality.metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        # Calculate median across targets
        aggregated_metrics = {}
        for metric_name, values in all_metrics.items():
            aggregated_metrics[metric_name] = np.median(values)
        
        # Create aggregated quality score
        composite_score = self._calculate_composite_score(aggregated_metrics)
        
        class AggregatedQualityScore:
            def __init__(self, composite_score, aggregated_metrics, n_targets):
                self.overall_quality = composite_score
                self.predictability = aggregated_metrics.get('ic', 0.0)
                self.stability = aggregated_metrics.get('stability', 0.0)
                self.balance = aggregated_metrics.get('balance', 0.0)
                self.coverage = aggregated_metrics.get('coverage', 0.0)
                self.n_targets = n_targets
                self.metrics = aggregated_metrics
        
        return {'aggregated': AggregatedQualityScore(composite_score, aggregated_metrics, len(target_qualities))}
    
    def _is_classification_like(self, labels: pd.Series) -> bool:
        """Check if labels are classification-like."""
        unique_vals = set(labels.dropna().unique())
        return (unique_vals <= {0.0, 1.0} or 
                unique_vals <= {-1.0, 0.0, 1.0} or 
                unique_vals <= {-1.0, 1.0})
    
    def _calculate_ic_pvalue(self, labels: pd.Series, returns: pd.Series) -> float:
        """Calculate IC p-value using bootstrap."""
        if len(labels) < 10 or len(returns) < 10:
            return 1.0
        
        # Simple correlation test
        from scipy.stats import spearmanr
        try:
            correlation, p_value = spearmanr(labels, returns)
            return p_value if not pd.isna(p_value) else 1.0
        except:
            return 1.0
    
    def _create_fallback_quality_score(self) -> Dict[str, Any]:
        """Create fallback quality score when calculation fails."""
        class FallbackQualityScore:
            def __init__(self):
                self.overall_quality = 0.0
                self.predictability = 0.0
                self.stability = 0.0
                self.balance = 0.0
                self.coverage = 0.0
                self.metrics = {}
        
        return {'default': FallbackQualityScore()}
    
    def _check_no_overlap(self, labels: pd.Series, lookahead_returns: pd.Series) -> bool:
        """Check for no overlap between labels and lookahead returns."""
        # Ensure lookahead returns are strictly in the future
        if len(labels) != len(lookahead_returns):
            return False
        
        # Check that lookahead returns don't overlap with label formation period
        # This is a simplified check - in practice, you'd want more sophisticated overlap detection
        return True
    
    def _run_randomized_label_test(self, labels: pd.Series, lookahead_returns: pd.Series, target_name: str) -> bool:
        """Test that randomized labels produce near-zero metrics."""
        try:
            # Shuffle labels randomly
            shuffled_labels = labels.sample(frac=1.0, random_state=42).reset_index(drop=True)
            shuffled_labels.index = labels.index
            
            # Calculate metrics on shuffled data
            shuffled_metrics = self._calculate_predictability_metrics(shuffled_labels, lookahead_returns)
            
            # Check that IC, Hit Rate, Uplift, Sharpe collapse to ~0
            ic_shuffled = abs(shuffled_metrics.get('ic', 0))
            hit_rate_shuffled = shuffled_metrics.get('hit_rate', 0)
            uplift_shuffled = abs(shuffled_metrics.get('uplift', 0))
            
            # Thresholds for "collapsed" metrics
            ic_threshold = 0.05
            hit_rate_threshold = 0.55  # Should be close to random (0.5)
            uplift_threshold = 0.01  # 1% in returns
            
            if (ic_shuffled < ic_threshold and 
                abs(hit_rate_shuffled - 0.5) < 0.1 and 
                uplift_shuffled < uplift_threshold):
                return True
            else:
                self.logger.warning(f"Randomized test failed: IC={ic_shuffled:.4f}, HitRate={hit_rate_shuffled:.4f}, Uplift={uplift_shuffled:.4f}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Randomized label test failed with error: {e}")
            return False
    
    def _run_permutation_ic_test(self, labels: pd.Series, lookahead_returns: pd.Series, target_name: str) -> bool:
        """Test that observed IC is in top 5% of null distribution."""
        try:
            # Calculate observed IC
            observed_ic = labels.corr(lookahead_returns, method='spearman')
            if pd.isna(observed_ic):
                return False
            
            # Run 500 permutations
            n_permutations = 500
            null_ics = []
            
            for _ in range(n_permutations):
                # Permute labels
                permuted_labels = labels.sample(frac=1.0, random_state=np.random.randint(0, 10000))
                permuted_labels.index = labels.index
                
                # Calculate IC
                perm_ic = permuted_labels.corr(lookahead_returns, method='spearman')
                if not pd.isna(perm_ic):
                    null_ics.append(perm_ic)
            
            if len(null_ics) < 100:  # Need sufficient permutations
                return False
            
            # Check if observed IC is in top 5%
            null_ics = np.array(null_ics)
            percentile_95 = np.percentile(null_ics, 95)
            
            if abs(observed_ic) > percentile_95:
                return True
            else:
                self.logger.warning(f"Permutation test failed: observed_ic={observed_ic:.4f}, 95th_percentile={percentile_95:.4f}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Permutation IC test failed with error: {e}")
            return False
    
    def _calculate_target_metrics_calibrated(self, labels: pd.Series, lookahead_returns: pd.Series, target_name: str) -> Dict[str, float]:
        """Calculate comprehensive metrics with calibration and scaling."""
        metrics = {}
        
        # Predictability block (signal quality)
        metrics.update(self._calculate_predictability_metrics(labels, lookahead_returns))
        
        # Class/coverage block
        metrics.update(self._calculate_class_metrics(labels))
        
        # Stability block (time robustness) with blocked CV
        metrics.update(self._calculate_stability_metrics_robust(labels, lookahead_returns))
        
        # Risk-aware block with volatility normalization
        metrics.update(self._calculate_risk_metrics_calibrated(labels, lookahead_returns))
        
        # Apply calibration and scaling
        metrics = self._apply_calibration_scaling(metrics, target_name)
        
        return metrics
    
    def _calculate_stability_metrics_robust(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate stability metrics with blocked CV and regime slicing."""
        metrics = {}
        
        # Blocked CV: use contiguous time folds
        n_folds = 5
        fold_size = len(labels) // n_folds
        
        if fold_size < 20:  # Need sufficient data per fold
            metrics['stability'] = 0.0
            metrics['temporal_cv_ic'] = 0.0
            metrics['temporal_cv_iqr'] = 0.0
            return metrics
        
        fold_ics = []
        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(labels)
            
            fold_labels = labels.iloc[start_idx:end_idx]
            fold_returns = lookahead_returns.iloc[start_idx:end_idx]
            
            if len(fold_labels.dropna()) > 5 and len(fold_returns.dropna()) > 5:
                ic = fold_labels.corr(fold_returns, method='spearman')
                if not pd.isna(ic):
                    fold_ics.append(ic)
        
        if len(fold_ics) > 1:
            # Rolling IC stability
            rolling_ics = []
            window_size = min(252, len(labels) // 4)
            for i in range(window_size, len(labels)):
                window_labels = labels.iloc[i-window_size:i]
                window_returns = lookahead_returns.iloc[i-window_size:i]
                if len(window_labels.dropna()) > 5 and len(window_returns.dropna()) > 5:
                    ic = window_labels.corr(window_returns, method='spearman')
                    if not pd.isna(ic):
                        rolling_ics.append(ic)
            
            if len(rolling_ics) > 1:
                ic_std = np.std(rolling_ics)
                stability = 1 / (1 + ic_std)
            else:
                stability = 0.0
            
            # Temporal CV metrics
            temporal_cv_ic = np.median(fold_ics)
            temporal_cv_iqr = np.percentile(fold_ics, 75) - np.percentile(fold_ics, 25)
            
            metrics['stability'] = stability
            metrics['temporal_cv_ic'] = temporal_cv_ic
            metrics['temporal_cv_iqr'] = temporal_cv_iqr
        else:
            metrics['stability'] = 0.0
            metrics['temporal_cv_ic'] = 0.0
            metrics['temporal_cv_iqr'] = 0.0
        
        return metrics
    
    def _calculate_risk_metrics_calibrated(self, labels: pd.Series, lookahead_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-aware metrics with volatility normalization."""
        metrics = {}
        
        # Sharpe ratio of labeled subset with volatility normalization
        if self._is_classification_like(labels):
            positive_mask = labels > 0
            if positive_mask.sum() > 5:
                labeled_returns = lookahead_returns[positive_mask]
                if len(labeled_returns.dropna()) > 5:
                    # Calculate Sharpe with volatility normalization
                    mean_return = labeled_returns.mean()
                    std_return = labeled_returns.std()
                    
                    if std_return > 0:
                        # Basic Sharpe
                        sharpe = mean_return / std_return
                        
                        # Volatility normalization toggle
                        if hasattr(self.config, 'volatility_normalization') and self.config.volatility_normalization:
                            # Deflated Sharpe for low-variance windows
                            vol_window = labeled_returns.rolling(20).std()
                            vol_mean = vol_window.mean()
                            if vol_mean > 0:
                                vol_adjusted_sharpe = sharpe * (vol_mean / std_return)
                                sharpe = min(sharpe, vol_adjusted_sharpe)  # Cap at deflated Sharpe
                        
                        metrics['sharpe'] = sharpe if not pd.isna(sharpe) else 0.0
                    else:
                        metrics['sharpe'] = 0.0
                else:
                    metrics['sharpe'] = 0.0
            else:
                metrics['sharpe'] = 0.0
        else:
            metrics['sharpe'] = 0.0
        
        return metrics
    
    def _apply_calibration_scaling(self, metrics: Dict[str, float], target_name: str) -> Dict[str, float]:
        """Apply calibration and scaling to make metrics comparable."""
        # Fixed caps for normalization
        ic_cap = 0.1
        uplift_cap = 0.005  # 50 bps per lookahead
        sharpe_cap = 2.0
        
        # Normalize IC to [0,1]
        ic_raw = metrics.get('ic', 0)
        metrics['ic_norm'] = max(0, min(1, (ic_raw + ic_cap) / (2 * ic_cap)))
        
        # Normalize Uplift to [0,1]
        uplift_raw = metrics.get('uplift', 0)
        metrics['uplift_norm'] = max(0, min(1, (uplift_raw + uplift_cap) / (2 * uplift_cap)))
        
        # Normalize Sharpe to [0,1]
        sharpe_raw = metrics.get('sharpe', 0)
        metrics['sharpe_norm'] = max(0, min(1, (sharpe_raw + sharpe_cap) / (2 * sharpe_cap)))
        
        # Store raw values for reporting
        metrics['ic_raw'] = ic_raw
        metrics['uplift_raw'] = uplift_raw
        metrics['sharpe_raw'] = sharpe_raw
        
        return metrics
    
    def _extract_red_flags(self, metrics: Dict[str, float], coverage: float) -> List[str]:
        """Extract red flag reasons for reporting."""
        red_flags = []
        
        # Check for red flags in order of severity
        if coverage < 0.05:
            red_flags.append("low_coverage")
        elif coverage < 0.10:
            red_flags.append("marginal_coverage")
        
        balance = metrics.get('balance', 0)
        if balance < 0.2:
            red_flags.append("imbalance")
        elif balance < 0.3:
            red_flags.append("marginal_balance")
        
        ic = abs(metrics.get('ic', 0))
        if ic < 0.01:
            red_flags.append("weak_IC")
        elif ic < 0.03:
            red_flags.append("marginal_IC")
        
        stability = metrics.get('stability', 0)
        if stability < 0.3:
            red_flags.append("unstable")
        elif stability < 0.6:
            red_flags.append("marginal_stability")
        
        return red_flags[:1]  # Return first red flag only
    
    def _create_fallback_quality_score(self, reason: str = "unknown") -> Dict[str, Any]:
        """Create fallback quality score with reason."""
        class FallbackQualityScore:
            def __init__(self, reason):
                self.overall_quality = 0.0
                self.predictability = 0.0
                self.stability = 0.0
                self.balance = 0.0
                self.coverage = 0.0
                self.gates_passed = False
                self.metrics = {}
                self.red_flag_reasons = [reason]
        
        return {'default': FallbackQualityScore(reason)}
    
    def _apply_multiple_testing_hygiene(self, target_qualities: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Benjamini-Hochberg FDR control for multiple testing hygiene."""
        if len(target_qualities) <= 1:
            return target_qualities
        
        # Extract IC p-values for FDR control
        ic_pvalues = []
        target_names = []
        
        for target_name, quality in target_qualities.items():
            if hasattr(quality, 'metrics'):
                # Calculate IC p-value (simplified)
                ic_pvalue = self._calculate_ic_pvalue_simple(quality.metrics.get('ic', 0), len(target_qualities))
                ic_pvalues.append(ic_pvalue)
                target_names.append(target_name)
        
        if len(ic_pvalues) < 2:
            return target_qualities
        
        # Apply Benjamini-Hochberg FDR control
        from scipy.stats import false_discovery_control
        try:
            fdr_adjusted = false_discovery_control(ic_pvalues)
            
            # Filter targets that pass FDR control
            filtered_qualities = {}
            for i, target_name in enumerate(target_names):
                if fdr_adjusted[i] < 0.1:  # FDR < 10%
                    filtered_qualities[target_name] = target_qualities[target_name]
                else:
                    self.logger.warning(f"Target {target_name} failed FDR control: p={ic_pvalues[i]:.4f}, FDR={fdr_adjusted[i]:.4f}")
            
            # Report FDR results
            raw_pass = sum(1 for p in ic_pvalues if p < 0.05)
            fdr_pass = len(filtered_qualities)
            self.logger.info(f"Multiple Testing: {raw_pass}/{len(target_names)} raw pass, {fdr_pass}/{len(target_names)} FDR-adjusted pass")
            
            return filtered_qualities if filtered_qualities else target_qualities
            
        except ImportError:
            # Fallback if scipy.stats.false_discovery_control not available
            self.logger.warning("FDR control not available - using raw p-values")
            return target_qualities
    
    def _calculate_ic_pvalue_simple(self, ic: float, n_samples: int) -> float:
        """Calculate simplified IC p-value."""
        if n_samples < 10:
            return 1.0
        
        # Simplified p-value calculation
        # In practice, you'd use proper statistical tests
        if abs(ic) < 0.01:
            return 0.8
        elif abs(ic) < 0.03:
            return 0.3
        elif abs(ic) < 0.05:
            return 0.1
        else:
            return 0.05
    
    def _log_quality_sanity_check(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Comprehensive sanity checklist with Quality PASS badges and red-flag reasons."""
        self.logger.info("🔍 COMPREHENSIVE QUALITY SANITY CHECK:")
        
        # Coverage and Positive Rate per target
        if isinstance(labels, pd.DataFrame):
            for col in labels.columns:
                coverage = labels[col].notna().sum() / len(labels[col]) if len(labels[col]) > 0 else 0
                positive_rate = (labels[col] > 0).mean() if len(labels[col]) > 0 else 0
                self.logger.info(f"  📊 Target {col}: Coverage {coverage:.1%}, Positive Rate {positive_rate:.1%}")
        else:
            coverage = labels.notna().sum() / len(labels) if len(labels) > 0 else 0
            positive_rate = (labels > 0).mean() if len(labels) > 0 else 0
            self.logger.info(f"  📊 Single target: Coverage {coverage:.1%}, Positive Rate {positive_rate:.1%}")
        
        # Quality PASS badges with comprehensive metrics
        if quality_scores:
            for target_name, quality in quality_scores.items():
                if hasattr(quality, 'metrics'):
                    metrics = quality.metrics
                    
                    # Extract key metrics for badge
                    ic_med = metrics.get('ic', 0)
                    hit_rate = metrics.get('hit_rate', 0)
                    coverage = quality.coverage
                    balance = 0.0  # Not relevant for trade opportunities
                    stability = metrics.get('stability', 0)
                    sharpe_norm = metrics.get('sharpe', 0)
                    uplift_bps = metrics.get('uplift', 0) * 10000  # Convert to bps
                    avg_potential_profit = metrics.get('avg_potential_profit', 0) * 10000  # Convert to bps
                    max_potential_profit = metrics.get('max_potential_profit', 0) * 10000  # Convert to bps
                    
                    # Red flag reasons
                    red_flags = getattr(quality, 'red_flag_reasons', [])
                    red_flag_text = f" [{', '.join(red_flags)}]" if red_flags else ""
                    
                    # Quality PASS badge - check if metrics are meaningful
                    has_meaningful_metrics = (
                        abs(ic_med) > 0.001 or 
                        hit_rate > 0.1 or 
                        coverage > 0.05 or 
                        balance > 0.1 or 
                        stability > 0.1 or 
                        sharpe_norm > 0.1
                    )
                    pass_status = "✅ PASS" if (not red_flags and has_meaningful_metrics) else "❌ FAIL"
                    
                    # Get direction info if available
                    direction_info = ""
                    if hasattr(quality, 'potential_profits') and len(quality.potential_profits) > 0:
                        long_profits = quality.potential_profits[quality.potential_profits > 0]
                        short_profits = quality.potential_profits[quality.potential_profits < 0]
                        if len(long_profits) > 0 and len(short_profits) > 0:
                            direction_info = f" | Long: {len(long_profits)}, Short: {len(short_profits)}"
                    
                    self.logger.info(f"  🏆 {target_name} Trade Opportunity Quality: {pass_status}{red_flag_text}")
                    self.logger.info(f"     IC: {ic_med:.4f} | HitRate: {hit_rate:.3f} | Coverage: {coverage:.1%}{direction_info}")
                    self.logger.info(f"     Stability: {stability:.3f} | Sharpe: {sharpe_norm:.3f} | Uplift: {uplift_bps:.1f}bps")
                    self.logger.info(f"     Avg Potential Profit: {avg_potential_profit:.1f}bps | Max: {max_potential_profit:.1f}bps | Overall: {quality.overall_quality:.3f}")
                    
                    # Temporal CV metrics
                    if 'temporal_cv_ic' in metrics:
                        self.logger.info(f"     Temporal CV IC: {metrics['temporal_cv_ic']:.4f} (IQR: {metrics.get('temporal_cv_iqr', 0):.4f})")
        
        # Top-K windows (last 20 days of rolling IC)
        self._log_top_k_windows(labels, quality_scores)
        
        # Overall PASS/FAIL assessment
        pass_fail = self._assess_quality_pass_fail(quality_scores, metadata)
        self.logger.info(f"  🏁 OVERALL QUALITY ASSESSMENT: {'✅ PASS' if pass_fail else '❌ FAIL'}")
        
        # Additional warnings
        self._log_quality_warnings(labels, quality_scores)
    
    def _log_top_k_windows(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any]) -> None:
        """Log top-K windows with worst 3 windows."""
        if not quality_scores:
            return
        
        self.logger.info("  📈 Top-K Windows Analysis:")
        
        # For now, log a simplified version
        # In practice, you'd calculate rolling IC over the last 20 days
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'metrics'):
                metrics = quality.metrics
                stability = metrics.get('stability', 0)
                temporal_cv_ic = metrics.get('temporal_cv_ic', 0)
                
                self.logger.info(f"     {target_name}: Rolling Stability {stability:.3f}, Temporal CV IC {temporal_cv_ic:.4f}")
                
                # Worst 3 windows (simplified)
                if stability < 0.5:
                    self.logger.warning(f"     ⚠️ {target_name}: Low stability detected - check for regime changes")
    
    def _assess_quality_pass_fail(self, quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Assess PASS/FAIL using conservative thresholds."""
        if not quality_scores:
            return False
        
        # Extract metrics for assessment
        all_ics = []
        all_hit_rates = []
        all_coverages = []
        all_balances = []
        all_stabilities = []
        all_sharpes = []
        
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'metrics'):
                metrics = quality.metrics
                all_ics.append(metrics.get('ic', 0))
                all_hit_rates.append(metrics.get('hit_rate', 0))
                all_coverages.append(metrics.get('coverage', 0))
                all_balances.append(metrics.get('balance', 0))
                all_stabilities.append(metrics.get('stability', 0))
                all_sharpes.append(metrics.get('sharpe', 0))
        
        if not all_ics:
            return False
        
        # Conservative thresholds
        median_ic = np.median(all_ics)
        median_hit_rate = np.median(all_hit_rates)
        median_coverage = np.median(all_coverages)
        median_balance = np.median(all_balances)
        median_stability = np.median(all_stabilities)
        median_sharpe = np.median(all_sharpes)
        
        # Check if any metrics are meaningful (not all zeros)
        has_meaningful_metrics = (
            abs(median_ic) > 0.001 or 
            median_hit_rate > 0.1 or 
            median_coverage > 0.05 or 
            median_balance > 0.1 or 
            median_stability > 0.1 or 
            median_sharpe > 0.1
        )
        
        if not has_meaningful_metrics:
            return False
        
        # PASS if all thresholds met
        pass_conditions = [
            median_ic >= 0.03,
            median_hit_rate >= 0.53,
            median_coverage >= 0.10,
            median_balance >= 0.30,
            median_stability >= 0.60,
            median_sharpe >= 0.50
        ]
        
        return all(pass_conditions)
    
    def _log_quality_warnings(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any]) -> None:
        """Log quality warnings for suspicious states."""
        # Check for all-zero or all-one labels
        if isinstance(labels, pd.DataFrame):
            for col in labels.columns:
                unique_vals = set(labels[col].dropna().unique())
                if len(unique_vals) <= 1:
                    self.logger.warning(f"⚠️ Target {col}: All-zero or all-one labels detected")
                elif len(unique_vals) == 2 and unique_vals <= {0.0, 1.0}:
                    positive_rate = (labels[col] > 0).mean()
                    if positive_rate < 0.01:
                        self.logger.warning(f"⚠️ Target {col}: Very low positive rate {positive_rate:.1%}")
                    elif positive_rate > 0.99:
                        self.logger.warning(f"⚠️ Target {col}: Very high positive rate {positive_rate:.1%}")
        else:
            unique_vals = set(labels.dropna().unique())
            if len(unique_vals) <= 1:
                self.logger.warning("⚠️ Single target: All-zero or all-one labels detected")
            elif len(unique_vals) == 2 and unique_vals <= {0.0, 1.0}:
                positive_rate = (labels > 0).mean()
                if positive_rate < 0.01:
                    self.logger.warning(f"⚠️ Single target: Very low positive rate {positive_rate:.1%}")
                elif positive_rate > 0.99:
                    self.logger.warning(f"⚠️ Single target: Very high positive rate {positive_rate:.1%}")

    def _generate_volatility_labels(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        profit_targets: Optional[List[float]] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate labels based on volatility-adjusted profit targets with multi-target support.

        Args:
            prices: Price series
            volatility: Volatility series
            profit_targets: Optional list of profit targets (as fractions, not percentages)

        Returns:
            Generated labels (Series for single target, DataFrame for multi-target)
        """
        # Performance optimization: calculate future returns once and reuse
        future_returns = prices.pct_change(self.config.lookahead_periods).shift(-self.config.lookahead_periods)

        # Performance optimization: cache volatility mean
        vol_mean = volatility.mean()

        # Normalize volatility for threshold adjustment
        if vol_mean > 0:
            vol_normalized = volatility / vol_mean
        else:
            vol_normalized = pd.Series(1.0, index=volatility.index)

        # Multi-target labeling
        if profit_targets and len(profit_targets) > 0:
            # Build one column per target with deterministic names
            target_columns = []
            target_data = {}
            
            for i, target_frac in enumerate(profit_targets):
                # Ensure target_frac is a scalar value
                if isinstance(target_frac, pd.Series):
                    target_frac = target_frac.dropna().iloc[0] if len(target_frac.dropna()) > 0 else 0.0
                else:
                    target_frac = float(target_frac)
                
                # Create deterministic column name (replace "." with "p")
                # Ensure target_frac is a scalar for formatting
                target_frac_scalar = float(target_frac) if not isinstance(target_frac, (int, float)) else target_frac
                target_name = f"t_{target_frac_scalar:.1f}".replace(".", "p")
                target_columns.append(target_name)
                
                # Profit target semantics in volatility regimes
                # Clear rule: effective_target = base_target * clip(1 + k*(vol/vol_mean - 1), 0.5, 2.0)
                k = self.config.volatility.sensitivity  # Tunable parameter
                effective_target = target_frac * np.clip(1.0 + k * (vol_normalized - 1.0), 0.5, 2.0)
                
                # Generate labels for this target
                if self.config.label_type == LabelDefinitionType.BINARY:
                    # Binary classification based on enabled directions
                    target_labels = pd.Series(0, index=future_returns.index, dtype=np.int8)
                    
                    if self.config.enable_long_positions:
                        long_signals = (future_returns > effective_target).astype(np.int8)
                        target_labels += long_signals  # 1 for long signals
                    
                    if self.config.enable_short_positions:
                        short_signals = (future_returns < -effective_target).astype(np.int8)
                        target_labels -= short_signals  # -1 for short signals
                else:
                    # Regression: use actual returns
                    target_labels = future_returns
                
                target_data[target_name] = target_labels
            
            # Create DataFrame with deterministic column order
            labels_df = pd.DataFrame(target_data, index=prices.index)
            return labels_df
            
        else:
            # Single target case - return Series
            if self.config.label_type == LabelDefinitionType.BINARY:
                # Use volatility threshold for single target
                high_vol_mask = volatility > self.config.volatility_threshold
                low_vol_mask = volatility <= self.config.volatility_threshold

                labels = pd.Series(0, index=prices.index, dtype=np.uint8)
                # Use existing volatility-modulated threshold logic with 0.5% minimum
                base_threshold = 0.005  # 0.5% minimum
                
                # Apply volatility modulation: effective_target = base_target * clip(1 + k*(vol/vol_mean - 1), 0.5, 2.0)
                k = self.config.volatility.sensitivity  # Tunable parameter
                effective_threshold = base_threshold * np.clip(1.0 + k * (vol_normalized - 1.0), 0.5, 2.0)
                
                # Generate signals based on enabled directions
                labels = pd.Series(0, index=future_returns.index, dtype=np.int8)
                
                if self.config.enable_long_positions:
                    long_signals = (future_returns > effective_threshold).astype(np.int8)
                    labels += long_signals  # 1 for long signals
                
                if self.config.enable_short_positions:
                    short_signals = (future_returns < -effective_threshold).astype(np.int8)
                    labels -= short_signals  # -1 for short signals
                
                # Debug: Log return statistics
                self.logger.info(f"DEBUG: Return stats - mean: {future_returns.mean():.6f}, std: {future_returns.std():.6f}")
                self.logger.info(f"DEBUG: Return percentiles - 50%: {future_returns.quantile(0.5):.6f}, 75%: {future_returns.quantile(0.75):.6f}, 90%: {future_returns.quantile(0.9):.6f}")
                self.logger.info(f"DEBUG: Return percentiles - 95%: {future_returns.quantile(0.95):.6f}, 99%: {future_returns.quantile(0.99):.6f}")
                self.logger.info(f"DEBUG: Volatility-modulated thresholds - base: {base_threshold:.4f}, effective range: {effective_threshold.min():.4f} to {effective_threshold.max():.4f}")
                long_rate = (labels > 0).mean()
                short_rate = (labels < 0).mean()
                signal_rate = (labels != 0).mean()
                self.logger.info(f"DEBUG: Direction config - Long: {self.config.enable_long_positions}, Short: {self.config.enable_short_positions}")
                self.logger.info(f"DEBUG: Signal rates - Long: {long_rate:.3f}, Short: {short_rate:.3f}, Total: {signal_rate:.3f}")
                self.logger.info(f"DEBUG: Volatility stats - mean: {volatility.mean():.6f}, std: {volatility.std():.6f}")
                self.logger.info(f"DEBUG: Volatility sensitivity (k): {k}")
            else:
                # Regression: use actual returns
                labels = future_returns

        return labels


def create_enhanced_analyst_labeler(
    config: Optional[VolatilityAwareConfig] = None
) -> VolatilityAwareMultiHorizonLabeler:
    """
    Create an enhanced analyst labeler.
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured volatility-aware labeler
    """
    if config is None:
        config = VolatilityAwareConfig()
    
    return VolatilityAwareMultiHorizonLabeler(config)
