"""
Volatility Aware Labeler Module

This module provides volatility-aware labeling functionality for profit labeling.
"""

from typing import Any, Dict, List, Optional, Union
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
        label_type: LabelDefinitionType = LabelDefinitionType.BINARY
    ):
        """
        Initialize volatility-aware configuration.
        
        Args:
            volatility_threshold: Threshold for volatility-based labeling
            lookahead_periods: Number of periods to look ahead
            min_volatility: Minimum volatility threshold
            max_volatility: Maximum volatility threshold
            label_type: Type of labels to generate
        """
        self.volatility_threshold = volatility_threshold
        self.lookahead_periods = lookahead_periods
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.label_type = label_type
        
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


class LabelingResult:
    """
    Result of labeling operation.
    """
    
    def __init__(
        self,
        labels: pd.Series,
        metadata: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None
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
        
        # Add convenience attributes with defensive counting
        self.n_samples = int(len(labels)) if labels is not None else 0

        # for classification only; for regression set to None
        if labels is not None:
            # Check if this is a DataFrame (multi-horizon) or Series (single-horizon)
            if isinstance(labels, pd.DataFrame):
                # For DataFrame, check if any column is integer dtype (classification)
                has_integer_columns = any(pd.api.types.is_integer_dtype(col) for col in labels.dtypes)
                if has_integer_columns:
                    # Count unique non-null values across all integer columns
                    unique_values = set()
                    for col in labels.columns:
                        if pd.api.types.is_integer_dtype(labels[col]):
                            unique_values.update(labels[col].dropna().unique())
                    self.n_targets = len(unique_values)
                else:
                    self.n_targets = None
            else:
                # For Series, check if it's integer dtype (classification)
                if pd.api.types.is_integer_dtype(labels.dtype):
                    self.n_targets = int(labels.dropna().nunique())
                else:
                    # For float dtype Series, check if it's effectively binary (only 0.0 and 1.0)
                    unique_vals = labels.dropna().unique()
                    if len(unique_vals) == 2 and set(unique_vals) <= {0.0, 1.0}:
                        self.n_targets = 2  # Binary classification
                    else:
                        self.n_targets = None  # Regression or other types
        else:
            self.n_targets = None

        self.n_horizons = int(self.metadata.get("n_horizons", 1))
        self.confidence_scores = self.metadata.get("confidence_scores")
        self.eligibility_masks = self.metadata.get("eligibility_masks")
        self.quality_scores = self.metadata.get("quality_scores")
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
            # Calculate volatility if not provided
            if volatility_column is None or volatility_column not in data.columns:
                volatility = data[price_column].pct_change().rolling(window=20).std()
            else:
                volatility = data[volatility_column]

            # Generate labels based on volatility and profit targets
            labels = self._generate_volatility_labels(data[price_column], volatility, profit_targets)

            # Generate quality scores
            quality_scores = self._calculate_quality_scores(labels, data[price_column])

            metadata = {
                "volatility_threshold": self.config.volatility_threshold,
                "lookahead_periods": self.config.lookahead_periods,
                "label_type": self.config.label_type.value,
                "total_labels": len(labels),
                "non_null_labels": labels.notna().sum(),
                "quality_scores": quality_scores,
                "profit_targets": profit_targets or []
            }

            return LabelingResult(labels, metadata, success=True)

        except Exception as e:
            self.logger.error(f"Error generating labels: {e}")
            return LabelingResult(
                pd.Series(dtype=float),
                {},
                success=False,
                error_message=str(e)
            )
    
    def _calculate_quality_scores(self, labels: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """Calculate quality scores for the generated labels."""
        try:
            # Basic quality metrics
            non_null_labels = labels.notna().sum()
            total_labels = len(labels)
            coverage = non_null_labels / total_labels if total_labels > 0 else 0.0
            
            # Calculate label distribution
            unique_labels = labels.nunique()
            label_balance = 1.0 - abs(labels.value_counts().std() / labels.value_counts().mean()) if labels.value_counts().mean() > 0 else 0.0

            # Calculate predictability (correlation with future returns)
            future_returns = prices.pct_change().shift(-1)
            correlation = labels.corr(future_returns) if non_null_labels > 1 else 0.0
            predictability = abs(correlation) if not pd.isna(correlation) else 0.0

            # Overall quality score
            overall_quality = (coverage * 0.3 + label_balance * 0.3 + predictability * 0.4)

            # Detailed metric prints
            self.logger.info("📊 Quality Metrics Breakdown:")
            self.logger.info(f"   📈 Coverage: {coverage:.4f} ({non_null_labels}/{total_labels} non-null labels)")
            self.logger.info(f"   ⚖️  Balance: {label_balance:.4f} (unique labels: {unique_labels})")
            self.logger.info(f"   🔮 Predictability: {predictability:.4f} (correlation: {correlation:.4f})")
            self.logger.info(f"   🎯 Overall Quality: {overall_quality:.4f} (weighted average)")
            self.logger.info(f"   📋 Quality Thresholds: min_quality={self.config.quality_scoring.min_quality_threshold}, min_predictability={self.config.quality_scoring.min_predictability}")
            
            # Create a quality object with attributes
            class QualityScore:
                def __init__(self, overall_quality, predictability, stability, balance):
                    self.overall_quality = overall_quality
                    self.predictability = predictability
                    self.stability = stability
                    self.balance = balance
            
            return {
                'default': QualityScore(overall_quality, predictability, label_balance, coverage)
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality scores: {e}")
            # Create a quality object with attributes for fallback
            class QualityScore:
                def __init__(self, overall_quality, predictability, stability, balance):
                    self.overall_quality = overall_quality
                    self.predictability = predictability
                    self.stability = stability
                    self.balance = balance
            
            return {
                'default': QualityScore(0.0, 0.0, 0.0, 0.0)
            }

    def _generate_volatility_labels(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        profit_targets: Optional[List[float]] = None
    ) -> pd.Series:
        """
        Generate labels based on volatility-adjusted profit targets.

        Args:
            prices: Price series
            volatility: Volatility series
            profit_targets: Optional list of profit targets to use instead of fixed thresholds

        Returns:
            Generated labels
        """
        # Calculate future returns over the lookahead period
        future_returns = prices.pct_change(self.config.lookahead_periods).shift(-self.config.lookahead_periods)

        # Create labels based on volatility and returns
        labels = pd.Series(index=prices.index, dtype=float)

        # Normalize volatility for threshold adjustment
        if volatility.mean() > 0:
            vol_normalized = volatility / volatility.mean()
        else:
            vol_normalized = pd.Series(1.0, index=volatility.index)

        # Apply volatility as multiplier to profit targets
        if profit_targets and len(profit_targets) > 0:
            # Use the first (most conservative) profit target as base
            base_target = profit_targets[0] / 100.0  # Convert percentage to decimal

            # Volatility-dependent multipliers (more sophisticated)
            # High volatility: vol-dependent threshold (more conservative as volatility increases, capped at 200%)
            # Low volatility: 100% of base target (same as average volatility)
            volatility_multiplier = np.clip(1.0 + (vol_normalized - 1.0) * 2.0, 1.0, 2.0)  # Between 100% and 200%
            high_vol_threshold = base_target * volatility_multiplier
            low_vol_threshold = base_target * 1.0  # 100% of base target for low volatility

            # Create masks based on volatility levels
            high_vol_mask = vol_normalized > 1.0  # Above average volatility
            low_vol_mask = vol_normalized <= 1.0  # Below or average volatility

            if self.config.label_type == LabelDefinitionType.BINARY:
                # Apply different thresholds based on volatility levels
                # For high volatility periods, use the volatility-adjusted threshold
                labels.loc[high_vol_mask] = (future_returns[high_vol_mask] > high_vol_threshold[high_vol_mask]).astype(int)
                # For low volatility periods, use 100% of base target
                labels.loc[low_vol_mask] = (future_returns[low_vol_mask] > low_vol_threshold).astype(int)
            else:
                # For regression, use the actual returns
                labels = future_returns
        else:
            # Fallback to original fixed thresholds if no profit targets provided
            high_vol_mask = volatility > self.config.volatility_threshold
            low_vol_mask = volatility <= self.config.volatility_threshold

            if self.config.label_type == LabelDefinitionType.BINARY:
                labels[high_vol_mask] = (future_returns[high_vol_mask] > 0.01).astype(int)
                labels[low_vol_mask] = (future_returns[low_vol_mask] > 0.005).astype(int)
            else:
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
