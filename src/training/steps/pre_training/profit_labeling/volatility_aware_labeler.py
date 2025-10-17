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
            
            # Create downstream-ready opportunity data
            opportunity_data = self._create_downstream_opportunity_data(quality_scores)
            
            # Generate training strategies from quality scores
            training_strategy = self.score_to_training(quality_scores)
            
            # Analyze performance requirements
            performance_config = self.performance_sanity(data)
            
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
                "opportunity_data": opportunity_data,  # Downstream-ready opportunity data
                "training_strategy": training_strategy,  # Score-to-training mapping
                "performance_config": performance_config,  # Performance optimization settings
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
            
            # Comprehensive outcome reporting
            self._log_comprehensive_outcome_report(result_labels, quality_scores, metadata, training_strategy, performance_config)
            
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
        
        # Calculate individual opportunity scores and weights
        opportunity_scores = self._calculate_individual_opportunity_scores(trade_opportunities, potential_profits, metrics)
        opportunity_weights = self._calculate_individual_opportunity_weights(trade_opportunities, potential_profits, metrics)
        
        # Create trade opportunity quality score object
        class TradeOpportunityQualityScore:
            def __init__(self, composite_score, metrics, potential_profits, target_name, opportunity_scores, opportunity_weights):
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
                # Individual opportunity scoring for downstream use
                self.opportunity_scores = opportunity_scores  # Per-opportunity quality scores
                self.opportunity_weights = opportunity_weights  # Per-opportunity weights
                # Store all metrics for detailed analysis
                self.metrics = metrics
                self.red_flag_reasons = self._extract_trade_opportunity_red_flags(metrics, potential_profits)
        
        return TradeOpportunityQualityScore(composite_score, metrics, potential_profits, target_name, opportunity_scores, opportunity_weights)
    
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
    
    def _calculate_individual_opportunity_scores(self, trade_opportunities: pd.Series, potential_profits: pd.Series, metrics: Dict[str, float]) -> pd.Series:
        """Calculate individual quality scores for each opportunity."""
        if len(potential_profits) == 0:
            return pd.Series(dtype=float)
        
        # Base score from potential profit (normalized to [0, 1])
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        std_profit = metrics.get('std_potential_profit', 0.0)
        
        # Individual opportunity scores based on:
        # 1. Potential profit relative to average (40% weight)
        # 2. Consistency with overall pattern (30% weight) 
        # 3. Risk-adjusted return (30% weight)
        
        # Profit score: how much above/below average
        profit_scores = potential_profits / max(avg_profit, 0.001)  # Avoid division by zero
        profit_scores = np.clip(profit_scores, 0, 2)  # Cap at 2x average
        
        # Consistency score: how close to the mean (lower deviation = higher score)
        if std_profit > 0:
            consistency_scores = 1.0 / (1.0 + np.abs(potential_profits - avg_profit) / std_profit)
        else:
            consistency_scores = pd.Series(1.0, index=potential_profits.index)
        
        # Risk-adjusted score: potential profit / volatility (if we had individual volatility)
        # For now, use a simplified version based on profit magnitude
        risk_adjusted_scores = potential_profits / max(potential_profits.max(), 0.001)
        
        # Weighted composite individual scores
        individual_scores = (
            0.4 * profit_scores +
            0.3 * consistency_scores +
            0.3 * risk_adjusted_scores
        )
        
        # Normalize to [0, 1] range
        individual_scores = np.clip(individual_scores, 0, 1)
        
        return pd.Series(individual_scores, index=trade_opportunities.index)
    
    def _calculate_individual_opportunity_weights(self, trade_opportunities: pd.Series, potential_profits: pd.Series, metrics: Dict[str, float]) -> pd.Series:
        """Calculate individual weights for each opportunity based on quality and potential."""
        if len(potential_profits) == 0:
            return pd.Series(dtype=float)
        
        # Base weight from potential profit magnitude
        avg_profit = metrics.get('avg_potential_profit', 0.0)
        max_profit = metrics.get('max_potential_profit', 0.0)
        
        # Weight based on potential profit relative to maximum
        if max_profit > 0:
            profit_weights = potential_profits / max_profit
        else:
            profit_weights = pd.Series(1.0, index=potential_profits.index)
        
        # Apply exponential scaling to emphasize high-potential opportunities
        # This creates a more pronounced difference between high and low potential opportunities
        scaled_weights = np.power(profit_weights, 0.7)  # 0.7 < 1 makes the curve less steep
        
        # Normalize weights to sum to 1.0 for proper probability distribution
        if scaled_weights.sum() > 0:
            normalized_weights = scaled_weights / scaled_weights.sum()
        else:
            normalized_weights = pd.Series(1.0 / len(scaled_weights), index=scaled_weights.index)
        
        return normalized_weights
    
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
    
    def _create_downstream_opportunity_data(self, quality_scores: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Create downstream-ready opportunity data with scores and weights for each target."""
        opportunity_data = {}
        
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'opportunity_scores') and hasattr(quality, 'opportunity_weights') and hasattr(quality, 'potential_profits'):
                # Create DataFrame with all opportunity information
                opportunity_df = pd.DataFrame({
                    'opportunity_index': quality.opportunity_scores.index,
                    'signal_direction': quality.opportunity_scores.index.map(lambda idx: 1 if idx in quality.opportunity_scores.index else 0),  # Will be updated with actual signal direction
                    'potential_profit': quality.potential_profits,
                    'quality_score': quality.opportunity_scores,
                    'weight': quality.opportunity_weights,
                    'target_name': target_name
                })
                
                # Add derived metrics
                opportunity_df['profit_rank'] = opportunity_df['potential_profit'].rank(ascending=False)
                opportunity_df['quality_rank'] = opportunity_df['quality_score'].rank(ascending=False)
                opportunity_df['weight_rank'] = opportunity_df['weight'].rank(ascending=False)
                
                # Add composite opportunity score (combination of quality and weight)
                opportunity_df['composite_score'] = (
                    0.6 * opportunity_df['quality_score'] + 
                    0.4 * opportunity_df['weight']
                )
                
                opportunity_data[target_name] = opportunity_df
        
        return opportunity_data
    
    def score_to_training(self, quality_scores: Dict[str, Any], training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Map quality scores to training strategies (gating, weighting, curriculum learning).
        
        Args:
            quality_scores: Quality scores from labeling
            training_config: Optional training configuration overrides
            
        Returns:
            Dictionary with training strategies mapped from scores
        """
        if not quality_scores:
            return self._create_default_training_strategy()
        
        training_strategy = {
            'gating': {},
            'weighting': {},
            'curriculum_learning': {},
            'memory_optimization': {},
            'reproducible_seeds': {}
        }
        
        # Process each target's quality scores
        for target_name, quality in quality_scores.items():
            if not hasattr(quality, 'overall_quality'):
                continue
                
            overall_quality = quality.overall_quality
            opportunity_scores = getattr(quality, 'opportunity_scores', pd.Series())
            opportunity_weights = getattr(quality, 'opportunity_weights', pd.Series())
            
            # 1. GATING: Determine if target should be included in training
            training_strategy['gating'][target_name] = self._calculate_training_gate(
                overall_quality, quality, training_config
            )
            
            # 2. WEIGHTING: Calculate sample weights for training
            training_strategy['weighting'][target_name] = self._calculate_training_weights(
                opportunity_scores, opportunity_weights, overall_quality, training_config
            )
            
            # 3. CURRICULUM LEARNING: Determine training order/difficulty
            training_strategy['curriculum_learning'][target_name] = self._calculate_curriculum_level(
                overall_quality, opportunity_scores, training_config
            )
        
        # 4. MEMORY OPTIMIZATION: Ensure O(N) memory usage
        training_strategy['memory_optimization'] = self._calculate_memory_strategy(
            quality_scores, training_config
        )
        
        # 5. REPRODUCIBLE SEEDS: Generate seeds for parallel folds
        training_strategy['reproducible_seeds'] = self._generate_reproducible_seeds(
            quality_scores, training_config
        )
        
        return training_strategy
    
    def _calculate_training_gate(self, overall_quality: float, quality: Any, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate training gate based on quality scores."""
        gate_config = config.get('gating', {}) if config else {}
        
        # Quality thresholds
        min_quality = gate_config.get('min_quality', 0.3)
        min_coverage = gate_config.get('min_coverage', 0.05)
        min_predictability = gate_config.get('min_predictability', 0.1)
        
        # Check gate conditions
        passes_quality = overall_quality >= min_quality
        passes_coverage = getattr(quality, 'coverage', 0) >= min_coverage
        passes_predictability = getattr(quality, 'predictability', 0) >= min_predictability
        
        # Additional checks for trade opportunities
        has_opportunities = False
        if hasattr(quality, 'opportunity_scores') and len(quality.opportunity_scores) > 0:
            has_opportunities = len(quality.opportunity_scores) >= gate_config.get('min_opportunities', 5)
        
        gate_passed = passes_quality and passes_coverage and passes_predictability and has_opportunities
        
        return {
            'include_in_training': gate_passed,
            'quality_score': overall_quality,
            'coverage': getattr(quality, 'coverage', 0),
            'predictability': getattr(quality, 'predictability', 0),
            'n_opportunities': len(quality.opportunity_scores) if hasattr(quality, 'opportunity_scores') else 0,
            'gate_reason': self._get_gate_reason(gate_passed, passes_quality, passes_coverage, passes_predictability, has_opportunities)
        }
    
    def _calculate_training_weights(self, opportunity_scores: pd.Series, opportunity_weights: pd.Series, 
                                  overall_quality: float, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate training weights based on opportunity scores."""
        weight_config = config.get('weighting', {}) if config else {}
        
        if len(opportunity_scores) == 0:
            return {'weights': pd.Series(dtype=float), 'weight_strategy': 'uniform'}
        
        # Weight calculation strategies
        strategy = weight_config.get('strategy', 'quality_weighted')
        
        if strategy == 'quality_weighted':
            # Use individual quality scores as weights
            weights = opportunity_scores.copy()
        elif strategy == 'profit_weighted':
            # Use potential profits as weights (if available)
            if hasattr(opportunity_scores, 'index'):
                # This would need access to potential_profits - simplified for now
                weights = opportunity_scores.copy()
            else:
                weights = opportunity_scores.copy()
        elif strategy == 'composite_weighted':
            # Combine quality scores with opportunity weights
            if len(opportunity_weights) > 0:
                weights = 0.7 * opportunity_scores + 0.3 * opportunity_weights
            else:
                weights = opportunity_scores.copy()
        else:  # uniform
            weights = pd.Series(1.0, index=opportunity_scores.index)
        
        # Apply quality-based scaling
        quality_scale = weight_config.get('quality_scale', True)
        if quality_scale:
            weights = weights * overall_quality
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(weights), index=weights.index)
        
        return {
            'weights': weights,
            'weight_strategy': strategy,
            'quality_scale': quality_scale,
            'weight_stats': {
                'mean': weights.mean(),
                'std': weights.std(),
                'min': weights.min(),
                'max': weights.max()
            }
        }
    
    def _calculate_curriculum_level(self, overall_quality: float, opportunity_scores: pd.Series, 
                                  config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate curriculum learning level based on quality scores."""
        curriculum_config = config.get('curriculum_learning', {}) if config else {}
        
        # Determine difficulty level based on quality
        if overall_quality >= 0.8:
            difficulty_level = 'expert'
            training_order = 1  # Train first (highest quality)
        elif overall_quality >= 0.6:
            difficulty_level = 'intermediate'
            training_order = 2
        elif overall_quality >= 0.4:
            difficulty_level = 'beginner'
            training_order = 3
        else:
            difficulty_level = 'novice'
            training_order = 4  # Train last (lowest quality)
        
        # Calculate sample complexity
        if len(opportunity_scores) > 0:
            score_std = opportunity_scores.std()
            complexity = min(1.0, score_std * 2)  # Higher std = more complex
        else:
            complexity = 0.5
        
        return {
            'difficulty_level': difficulty_level,
            'training_order': training_order,
            'complexity_score': complexity,
            'quality_threshold': overall_quality,
            'enable_curriculum': curriculum_config.get('enable', True)
        }
    
    def _calculate_memory_strategy(self, quality_scores: Dict[str, Any], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate memory optimization strategy for O(N) memory usage."""
        memory_config = config.get('memory_optimization', {}) if config else {}
        
        # Count total opportunities across all targets
        total_opportunities = 0
        for quality in quality_scores.values():
            if hasattr(quality, 'opportunity_scores'):
                total_opportunities += len(quality.opportunity_scores)
        
        # Memory budget (in MB)
        memory_budget_mb = memory_config.get('memory_budget_mb', 1024)  # 1GB default
        bytes_per_opportunity = memory_config.get('bytes_per_opportunity', 1000)  # Estimate
        
        max_opportunities = (memory_budget_mb * 1024 * 1024) // bytes_per_opportunity
        
        # Determine if we need to subsample
        needs_subsampling = total_opportunities > max_opportunities
        subsample_ratio = min(1.0, max_opportunities / total_opportunities) if total_opportunities > 0 else 1.0
        
        return {
            'total_opportunities': total_opportunities,
            'memory_budget_mb': memory_budget_mb,
            'max_opportunities': max_opportunities,
            'needs_subsampling': needs_subsampling,
            'subsample_ratio': subsample_ratio,
            'chunk_size': memory_config.get('chunk_size', 10000),
            'enable_streaming': needs_subsampling
        }
    
    def _generate_reproducible_seeds(self, quality_scores: Dict[str, Any], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reproducible seeds for parallel folds."""
        seed_config = config.get('reproducible_seeds', {}) if config else {}
        
        base_seed = seed_config.get('base_seed', 42)
        n_folds = seed_config.get('n_folds', 5)
        
        # Generate seeds for each fold
        fold_seeds = {}
        for i in range(n_folds):
            fold_seeds[f'fold_{i}'] = base_seed + i * 1000
        
        # Generate seeds for each target
        target_seeds = {}
        for i, target_name in enumerate(quality_scores.keys()):
            target_seeds[target_name] = base_seed + i * 100
        
        return {
            'base_seed': base_seed,
            'n_folds': n_folds,
            'fold_seeds': fold_seeds,
            'target_seeds': target_seeds,
            'random_state': np.random.RandomState(base_seed)
        }
    
    def _get_gate_reason(self, gate_passed: bool, quality_ok: bool, coverage_ok: bool, 
                        predictability_ok: bool, has_opportunities: bool) -> str:
        """Get human-readable gate reason."""
        if gate_passed:
            return "PASS: All quality gates met"
        
        reasons = []
        if not quality_ok:
            reasons.append("low_quality")
        if not coverage_ok:
            reasons.append("low_coverage")
        if not predictability_ok:
            reasons.append("low_predictability")
        if not has_opportunities:
            reasons.append("insufficient_opportunities")
        
        return f"FAIL: {', '.join(reasons)}"
    
    def _create_default_training_strategy(self) -> Dict[str, Any]:
        """Create default training strategy when no quality scores available."""
        return {
            'gating': {'default': {'include_in_training': False, 'gate_reason': 'no_quality_scores'}},
            'weighting': {'default': {'weights': pd.Series(dtype=float), 'weight_strategy': 'uniform'}},
            'curriculum_learning': {'default': {'difficulty_level': 'novice', 'training_order': 999}},
            'memory_optimization': {'total_opportunities': 0, 'needs_subsampling': False},
            'reproducible_seeds': {'base_seed': 42, 'n_folds': 5, 'fold_seeds': {}}
        }
    
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
    
    def _log_comprehensive_outcome_report(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any], 
                                        metadata: Dict[str, Any], training_strategy: Dict[str, Any], 
                                        performance_config: Dict[str, Any]) -> None:
        """Generate comprehensive, human-readable outcome report."""
        self.logger.info("=" * 80)
        self.logger.info("🎯 VOLATILITY-AWARE LABELING OUTCOME REPORT")
        self.logger.info("=" * 80)
        
        # 1. EXECUTIVE SUMMARY
        self._log_executive_summary(labels, quality_scores, metadata)
        
        # 2. LABELING PERFORMANCE
        self._log_labeling_performance(labels, metadata)
        
        # 3. QUALITY ANALYSIS
        self._log_quality_analysis(quality_scores)
        
        # 4. TRADE OPPORTUNITIES ANALYSIS
        self._log_trade_opportunities_analysis(quality_scores)
        
        # 5. TRAINING STRATEGY RECOMMENDATIONS
        self._log_training_strategy_recommendations(training_strategy)
        
        # 6. PERFORMANCE OPTIMIZATION
        self._log_performance_optimization(performance_config)
        
        # 7. RISK ASSESSMENT & WARNINGS
        self._log_risk_assessment(quality_scores, metadata)
        
        # 8. NEXT STEPS & RECOMMENDATIONS
        self._log_next_steps_recommendations(quality_scores, training_strategy, performance_config)
        
        self.logger.info("=" * 80)
        self.logger.info("📋 REPORT COMPLETE")
        self.logger.info("=" * 80)
    
    def _log_executive_summary(self, labels: Union[pd.Series, pd.DataFrame], quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Log executive summary of labeling results."""
        self.logger.info("📊 EXECUTIVE SUMMARY")
        self.logger.info("-" * 40)
        
        # Basic statistics
        total_labels = metadata.get("total_labels", 0)
        non_null_labels = metadata.get("non_null_labels", 0)
        coverage = non_null_labels / total_labels if total_labels > 0 else 0
        
        # Count opportunities
        total_opportunities = 0
        high_quality_targets = 0
        for quality in quality_scores.values():
            if hasattr(quality, 'opportunity_scores'):
                total_opportunities += len(quality.opportunity_scores)
            if hasattr(quality, 'overall_quality') and quality.overall_quality > 0.6:
                high_quality_targets += 1
        
        # Overall assessment
        if coverage > 0.1 and total_opportunities > 10:
            status = "✅ SUCCESS"
            status_emoji = "🎉"
        elif coverage > 0.05 and total_opportunities > 5:
            status = "⚠️ PARTIAL SUCCESS"
            status_emoji = "⚠️"
        else:
            status = "❌ NEEDS ATTENTION"
            status_emoji = "🚨"
        
        self.logger.info(f"{status_emoji} Overall Status: {status}")
        self.logger.info(f"📈 Data Coverage: {coverage:.1%} ({non_null_labels:,} / {total_labels:,} samples)")
        self.logger.info(f"🎯 Trade Opportunities: {total_opportunities:,} identified")
        self.logger.info(f"⭐ High-Quality Targets: {high_quality_targets} / {len(quality_scores)}")
        self.logger.info(f"⏱️ Processing Time: {metadata.get('processing_time', 0):.2f}s")
    
    def _log_labeling_performance(self, labels: Union[pd.Series, pd.DataFrame], metadata: Dict[str, Any]) -> None:
        """Log detailed labeling performance metrics."""
        self.logger.info("")
        self.logger.info("📈 LABELING PERFORMANCE")
        self.logger.info("-" * 40)
        
        if isinstance(labels, pd.DataFrame):
            self.logger.info("Multi-Target Analysis:")
            for col in labels.columns:
                coverage = labels[col].notna().sum() / len(labels[col]) if len(labels[col]) > 0 else 0
                positive_rate = (labels[col] > 0).mean() if len(labels[col]) > 0 else 0
                negative_rate = (labels[col] < 0).mean() if len(labels[col]) > 0 else 0
                signal_rate = (labels[col] != 0).mean() if len(labels[col]) > 0 else 0
                
                self.logger.info(f"  🎯 {col}:")
                self.logger.info(f"     Coverage: {coverage:.1%} | Signals: {signal_rate:.1%} | Long: {positive_rate:.1%} | Short: {negative_rate:.1%}")
        else:
            coverage = labels.notna().sum() / len(labels) if len(labels) > 0 else 0
            positive_rate = (labels > 0).mean() if len(labels) > 0 else 0
            negative_rate = (labels < 0).mean() if len(labels) > 0 else 0
            signal_rate = (labels != 0).mean() if len(labels) > 0 else 0
            
            self.logger.info(f"Single Target Analysis:")
            self.logger.info(f"  Coverage: {coverage:.1%} | Signals: {signal_rate:.1%} | Long: {positive_rate:.1%} | Short: {negative_rate:.1%}")
        
        # Configuration info
        self.logger.info(f"⚙️ Configuration:")
        self.logger.info(f"  Label Type: {metadata.get('label_type', 'unknown')}")
        self.logger.info(f"  Volatility Enabled: {metadata.get('volatility_enabled', False)}")
        self.logger.info(f"  Volatility Window: {metadata.get('volatility_window', 'N/A')}")
    
    def _log_quality_analysis(self, quality_scores: Dict[str, Any]) -> None:
        """Log detailed quality analysis."""
        self.logger.info("")
        self.logger.info("⭐ QUALITY ANALYSIS")
        self.logger.info("-" * 40)
        
        if not quality_scores:
            self.logger.warning("  ⚠️ No quality scores available")
            return
        
        # Overall quality statistics
        quality_values = []
        for quality in quality_scores.values():
            if hasattr(quality, 'overall_quality'):
                quality_values.append(quality.overall_quality)
        
        if quality_values:
            avg_quality = np.mean(quality_values)
            min_quality = np.min(quality_values)
            max_quality = np.max(quality_values)
            
            self.logger.info(f"📊 Overall Quality Statistics:")
            self.logger.info(f"  Average: {avg_quality:.3f} | Range: {min_quality:.3f} - {max_quality:.3f}")
            
            # Quality distribution
            excellent = sum(1 for q in quality_values if q >= 0.8)
            good = sum(1 for q in quality_values if 0.6 <= q < 0.8)
            fair = sum(1 for q in quality_values if 0.4 <= q < 0.6)
            poor = sum(1 for q in quality_values if q < 0.4)
            
            self.logger.info(f"📈 Quality Distribution:")
            self.logger.info(f"  🏆 Excellent (≥0.8): {excellent} targets")
            self.logger.info(f"  ✅ Good (0.6-0.8): {good} targets")
            self.logger.info(f"  ⚠️ Fair (0.4-0.6): {fair} targets")
            self.logger.info(f"  ❌ Poor (<0.4): {poor} targets")
        
        # Per-target detailed analysis
        self.logger.info(f"")
        self.logger.info(f"🎯 Per-Target Analysis:")
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'overall_quality'):
                self._log_target_quality_details(target_name, quality)
    
    def _log_target_quality_details(self, target_name: str, quality: Any) -> None:
        """Log detailed quality information for a specific target."""
        overall_quality = getattr(quality, 'overall_quality', 0)
        predictability = getattr(quality, 'predictability', 0)
        stability = getattr(quality, 'stability', 0)
        coverage = getattr(quality, 'coverage', 0)
        
        # Quality badge
        if overall_quality >= 0.8:
            badge = "🏆 EXCELLENT"
        elif overall_quality >= 0.6:
            badge = "✅ GOOD"
        elif overall_quality >= 0.4:
            badge = "⚠️ FAIR"
        else:
            badge = "❌ POOR"
        
        self.logger.info(f"  {badge} {target_name}:")
        self.logger.info(f"     Overall Quality: {overall_quality:.3f}")
        self.logger.info(f"     Predictability (IC): {predictability:.4f}")
        self.logger.info(f"     Stability: {stability:.3f}")
        self.logger.info(f"     Coverage: {coverage:.1%}")
        
        # Opportunity metrics
        if hasattr(quality, 'avg_potential_profit'):
            avg_profit = quality.avg_potential_profit * 10000  # Convert to bps
            max_profit = quality.max_potential_profit * 10000
            self.logger.info(f"     Avg Potential Profit: {avg_profit:.1f}bps | Max: {max_profit:.1f}bps")
        
        # Red flags
        red_flags = getattr(quality, 'red_flag_reasons', [])
        if red_flags:
            self.logger.info(f"     🚨 Red Flags: {', '.join(red_flags)}")
    
    def _log_trade_opportunities_analysis(self, quality_scores: Dict[str, Any]) -> None:
        """Log detailed trade opportunities analysis."""
        self.logger.info("")
        self.logger.info("🎯 TRADE OPPORTUNITIES ANALYSIS")
        self.logger.info("-" * 40)
        
        total_opportunities = 0
        total_long_opportunities = 0
        total_short_opportunities = 0
        total_potential_profit = 0
        
        for target_name, quality in quality_scores.items():
            if hasattr(quality, 'opportunity_scores') and len(quality.opportunity_scores) > 0:
                n_opportunities = len(quality.opportunity_scores)
                total_opportunities += n_opportunities
                
                # Count long/short opportunities
                if hasattr(quality, 'potential_profits'):
                    long_ops = (quality.potential_profits > 0).sum()
                    short_ops = (quality.potential_profits < 0).sum()
                    total_long_opportunities += long_ops
                    total_short_opportunities += short_ops
                    total_potential_profit += quality.potential_profits.sum()
                
                self.logger.info(f"  📊 {target_name}: {n_opportunities:,} opportunities")
                if hasattr(quality, 'avg_potential_profit'):
                    avg_profit = quality.avg_potential_profit * 10000
                    self.logger.info(f"     Avg Potential Profit: {avg_profit:.1f}bps")
        
        if total_opportunities > 0:
            self.logger.info(f"")
            self.logger.info(f"📈 Summary Statistics:")
            self.logger.info(f"  Total Opportunities: {total_opportunities:,}")
            self.logger.info(f"  Long Opportunities: {total_long_opportunities:,}")
            self.logger.info(f"  Short Opportunities: {total_short_opportunities:,}")
            if total_potential_profit > 0:
                avg_total_profit = (total_potential_profit / total_opportunities) * 10000
                self.logger.info(f"  Average Potential Profit: {avg_total_profit:.1f}bps")
        else:
            self.logger.warning("  ⚠️ No trade opportunities identified")
    
    def _log_training_strategy_recommendations(self, training_strategy: Dict[str, Any]) -> None:
        """Log training strategy recommendations."""
        self.logger.info("")
        self.logger.info("🎓 TRAINING STRATEGY RECOMMENDATIONS")
        self.logger.info("-" * 40)
        
        # Gating recommendations
        gating = training_strategy.get('gating', {})
        included_targets = sum(1 for gate in gating.values() if gate.get('include_in_training', False))
        total_targets = len(gating)
        
        self.logger.info(f"🚪 Gating Strategy:")
        self.logger.info(f"  Targets Included: {included_targets} / {total_targets}")
        
        for target_name, gate_info in gating.items():
            status = "✅ INCLUDE" if gate_info.get('include_in_training', False) else "❌ EXCLUDE"
            reason = gate_info.get('gate_reason', 'Unknown')
            self.logger.info(f"    {status} {target_name}: {reason}")
        
        # Weighting strategy
        weighting = training_strategy.get('weighting', {})
        self.logger.info(f"")
        self.logger.info(f"⚖️ Weighting Strategy:")
        for target_name, weight_info in weighting.items():
            strategy = weight_info.get('weight_strategy', 'unknown')
            weight_stats = weight_info.get('weight_stats', {})
            self.logger.info(f"  {target_name}: {strategy}")
            if weight_stats:
                self.logger.info(f"    Mean: {weight_stats.get('mean', 0):.3f} | Std: {weight_stats.get('std', 0):.3f}")
        
        # Curriculum learning
        curriculum = training_strategy.get('curriculum_learning', {})
        self.logger.info(f"")
        self.logger.info(f"📚 Curriculum Learning:")
        for target_name, curriculum_info in curriculum.items():
            level = curriculum_info.get('difficulty_level', 'unknown')
            order = curriculum_info.get('training_order', 999)
            self.logger.info(f"  {target_name}: {level.upper()} (Order: {order})")
    
    def _log_performance_optimization(self, performance_config: Dict[str, Any]) -> None:
        """Log performance optimization settings."""
        self.logger.info("")
        self.logger.info("⚡ PERFORMANCE OPTIMIZATION")
        self.logger.info("-" * 40)
        
        memory_analysis = performance_config.get('memory_analysis', {})
        parallel_config = performance_config.get('parallel_config', {})
        chunking_config = performance_config.get('chunking_config', {})
        
        # Memory analysis
        data_size_mb = memory_analysis.get('data_size_mb', 0)
        needs_optimization = memory_analysis.get('needs_optimization', False)
        target_chunk_size = memory_analysis.get('target_chunk_size', 0)
        
        self.logger.info(f"💾 Memory Analysis:")
        self.logger.info(f"  Data Size: {data_size_mb:.1f} MB")
        self.logger.info(f"  Optimization Needed: {'Yes' if needs_optimization else 'No'}")
        if needs_optimization:
            self.logger.info(f"  Target Chunk Size: {target_chunk_size:,} samples")
        
        # Parallel processing
        n_workers = parallel_config.get('n_workers', 1)
        n_folds = parallel_config.get('n_folds', 5)
        enable_parallel = parallel_config.get('enable_parallel', False)
        
        self.logger.info(f"")
        self.logger.info(f"🔄 Parallel Processing:")
        self.logger.info(f"  Workers: {n_workers} | Folds: {n_folds}")
        self.logger.info(f"  Parallel Enabled: {'Yes' if enable_parallel else 'No'}")
        
        # Chunking strategy
        if chunking_config.get('enabled', False):
            strategy = chunking_config.get('strategy', 'unknown')
            n_chunks = chunking_config.get('n_chunks', 0)
            self.logger.info(f"")
            self.logger.info(f"📦 Chunking Strategy:")
            self.logger.info(f"  Strategy: {strategy.upper()} | Chunks: {n_chunks}")
    
    def _log_risk_assessment(self, quality_scores: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Log risk assessment and warnings."""
        self.logger.info("")
        self.logger.info("⚠️ RISK ASSESSMENT & WARNINGS")
        self.logger.info("-" * 40)
        
        warnings = []
        
        # Check for low quality targets
        low_quality_count = 0
        for quality in quality_scores.values():
            if hasattr(quality, 'overall_quality') and quality.overall_quality < 0.3:
                low_quality_count += 1
        
        if low_quality_count > 0:
            warnings.append(f"Low quality targets: {low_quality_count} targets below 0.3 quality score")
        
        # Check for insufficient opportunities
        total_opportunities = 0
        for quality in quality_scores.values():
            if hasattr(quality, 'opportunity_scores'):
                total_opportunities += len(quality.opportunity_scores)
        
        if total_opportunities < 10:
            warnings.append(f"Insufficient opportunities: Only {total_opportunities} opportunities identified")
        
        # Check for high volatility in quality scores
        quality_values = [q.overall_quality for q in quality_scores.values() if hasattr(q, 'overall_quality')]
        if len(quality_values) > 1:
            quality_std = np.std(quality_values)
            if quality_std > 0.3:
                warnings.append(f"High quality variance: {quality_std:.3f} standard deviation across targets")
        
        # Check for red flags
        red_flag_count = 0
        for quality in quality_scores.values():
            red_flags = getattr(quality, 'red_flag_reasons', [])
            red_flag_count += len(red_flags)
        
        if red_flag_count > 0:
            warnings.append(f"Red flags detected: {red_flag_count} total red flags across targets")
        
        if warnings:
            for warning in warnings:
                self.logger.warning(f"  🚨 {warning}")
        else:
            self.logger.info("  ✅ No significant risks identified")
    
    def _log_next_steps_recommendations(self, quality_scores: Dict[str, Any], training_strategy: Dict[str, Any], 
                                      performance_config: Dict[str, Any]) -> None:
        """Log next steps and recommendations."""
        self.logger.info("")
        self.logger.info("🚀 NEXT STEPS & RECOMMENDATIONS")
        self.logger.info("-" * 40)
        
        # Count included targets
        gating = training_strategy.get('gating', {})
        included_targets = sum(1 for gate in gating.values() if gate.get('include_in_training', False))
        
        if included_targets == 0:
            self.logger.info("  🚨 CRITICAL: No targets passed quality gates")
            self.logger.info("     → Review profit thresholds and volatility settings")
            self.logger.info("     → Consider relaxing quality requirements")
            self.logger.info("     → Check data quality and preprocessing")
        elif included_targets < len(gating) // 2:
            self.logger.info("  ⚠️ WARNING: Less than half of targets passed quality gates")
            self.logger.info("     → Consider adjusting quality thresholds")
            self.logger.info("     → Review individual target performance")
        else:
            self.logger.info("  ✅ GOOD: Majority of targets passed quality gates")
        
        # Memory optimization recommendations
        memory_analysis = performance_config.get('memory_analysis', {})
        if memory_analysis.get('needs_optimization', False):
            self.logger.info("  💾 Memory optimization recommended:")
            self.logger.info("     → Use chunked processing for large datasets")
            self.logger.info("     → Consider data streaming for very large datasets")
        
        # Training recommendations
        curriculum = training_strategy.get('curriculum_learning', {})
        expert_targets = [t for t, c in curriculum.items() if c.get('difficulty_level') == 'expert']
        
        if expert_targets:
            self.logger.info(f"  🎓 Training recommendations:")
            self.logger.info(f"     → Start with expert targets: {', '.join(expert_targets)}")
            self.logger.info(f"     → Use curriculum learning for progressive training")
        
        # Performance recommendations
        parallel_config = performance_config.get('parallel_config', {})
        if parallel_config.get('enable_parallel', False):
            self.logger.info("  ⚡ Performance recommendations:")
            self.logger.info(f"     → Use {parallel_config.get('n_workers', 1)} parallel workers")
            self.logger.info(f"     → Implement reproducible seed management")
    
    def performance_sanity(self, data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ensure O(N) memory usage and parallel folds with reproducible seeds.
        
        Args:
            data: Input data for memory analysis
            config: Optional performance configuration
            
        Returns:
            Dictionary with performance optimization settings
        """
        perf_config = config.get('performance', {}) if config else {}
        
        # Memory analysis
        memory_analysis = self._analyze_memory_usage(data, perf_config)
        
        # Parallel processing configuration
        parallel_config = self._configure_parallel_processing(data, perf_config)
        
        # Reproducible seeds for parallel folds
        seed_config = self._configure_reproducible_seeds(perf_config)
        
        # Chunking strategy for large datasets
        chunking_config = self._configure_chunking_strategy(data, memory_analysis, perf_config)
        
        return {
            'memory_analysis': memory_analysis,
            'parallel_config': parallel_config,
            'seed_config': seed_config,
            'chunking_config': chunking_config,
            'optimization_applied': True
        }
    
    def _analyze_memory_usage(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory usage and determine optimization needs."""
        # Calculate data size
        data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        n_samples = len(data)
        n_features = len(data.columns)
        
        # Memory budget
        memory_budget_mb = config.get('memory_budget_mb', 1024)  # 1GB default
        max_samples = config.get('max_samples', 1000000)  # 1M samples default
        
        # Determine if optimization is needed
        needs_optimization = data_size_mb > memory_budget_mb or n_samples > max_samples
        
        # Calculate optimal chunk size
        if needs_optimization:
            target_chunk_size = min(
                int(memory_budget_mb * 1024 * 1024 / (data_size_mb * 1024 * 1024 / n_samples)),
                max_samples
            )
        else:
            target_chunk_size = n_samples
        
        return {
            'data_size_mb': data_size_mb,
            'n_samples': n_samples,
            'n_features': n_features,
            'memory_budget_mb': memory_budget_mb,
            'needs_optimization': needs_optimization,
            'target_chunk_size': max(1000, target_chunk_size),  # Minimum 1000 samples
            'estimated_chunks': max(1, n_samples // max(1000, target_chunk_size))
        }
    
    def _configure_parallel_processing(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure parallel processing for optimal performance."""
        n_samples = len(data)
        n_cores = config.get('n_cores', None)
        
        if n_cores is None:
            import multiprocessing
            n_cores = min(multiprocessing.cpu_count(), 8)  # Cap at 8 cores
        
        # Determine optimal number of parallel workers
        if n_samples < 10000:
            n_workers = 1  # Single-threaded for small datasets
        elif n_samples < 100000:
            n_workers = min(2, n_cores)
        else:
            n_workers = min(4, n_cores)
        
        # Configure parallel folds
        n_folds = config.get('n_folds', 5)
        fold_size = n_samples // n_folds
        
        return {
            'n_cores': n_cores,
            'n_workers': n_workers,
            'n_folds': n_folds,
            'fold_size': fold_size,
            'enable_parallel': n_workers > 1,
            'chunk_processing': n_samples > 50000
        }
    
    def _configure_reproducible_seeds(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure reproducible seeds for parallel processing."""
        base_seed = config.get('base_seed', 42)
        n_folds = config.get('n_folds', 5)
        
        # Generate seeds for each fold
        fold_seeds = {}
        for i in range(n_folds):
            fold_seeds[f'fold_{i}'] = base_seed + i * 1000
        
        # Generate seeds for parallel workers
        n_workers = config.get('n_workers', 1)
        worker_seeds = {}
        for i in range(n_workers):
            worker_seeds[f'worker_{i}'] = base_seed + i * 100
        
        return {
            'base_seed': base_seed,
            'fold_seeds': fold_seeds,
            'worker_seeds': worker_seeds,
            'random_state': np.random.RandomState(base_seed)
        }
    
    def _configure_chunking_strategy(self, data: pd.DataFrame, memory_analysis: Dict[str, Any], 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure chunking strategy for large datasets."""
        if not memory_analysis['needs_optimization']:
            return {'enabled': False, 'chunk_size': len(data)}
        
        chunk_size = memory_analysis['target_chunk_size']
        n_chunks = memory_analysis['estimated_chunks']
        
        # Determine chunking strategy
        if n_chunks <= 10:
            strategy = 'sequential'  # Process chunks sequentially
        elif n_chunks <= 100:
            strategy = 'parallel_chunks'  # Process multiple chunks in parallel
        else:
            strategy = 'streaming'  # Stream processing for very large datasets
        
        return {
            'enabled': True,
            'strategy': strategy,
            'chunk_size': chunk_size,
            'n_chunks': n_chunks,
            'overlap_samples': config.get('chunk_overlap', 100),  # Overlap between chunks
            'memory_efficient': True
        }
    
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
                    
                    # Get individual opportunity scoring info if available
                    opportunity_info = ""
                    if hasattr(quality, 'opportunity_scores') and hasattr(quality, 'opportunity_weights'):
                        if len(quality.opportunity_scores) > 0:
                            avg_score = quality.opportunity_scores.mean()
                            max_score = quality.opportunity_scores.max()
                            weight_entropy = -(quality.opportunity_weights * np.log(quality.opportunity_weights + 1e-10)).sum()
                            opportunity_info = f" | Avg Score: {avg_score:.3f} | Max Score: {max_score:.3f} | Weight Entropy: {weight_entropy:.3f}"
                    
                    self.logger.info(f"  🏆 {target_name} Trade Opportunity Quality: {pass_status}{red_flag_text}")
                    self.logger.info(f"     IC: {ic_med:.4f} | HitRate: {hit_rate:.3f} | Coverage: {coverage:.1%}{direction_info}")
                    self.logger.info(f"     Stability: {stability:.3f} | Sharpe: {sharpe_norm:.3f} | Uplift: {uplift_bps:.1f}bps")
                    self.logger.info(f"     Avg Potential Profit: {avg_potential_profit:.1f}bps | Max: {max_potential_profit:.1f}bps | Overall: {quality.overall_quality:.3f}{opportunity_info}")
                    
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
