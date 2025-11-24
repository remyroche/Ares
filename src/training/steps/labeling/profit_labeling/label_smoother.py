"""
Label Smoothing Module

Implements three-stage label smoothing for robust machine learning targets:
1. Classification/Probability smoothing (calibration)
2. Uncertainty-weighted shrinkage (sample reliability)
3. Causal EMA (temporal stabilization)

This module provides drop-in label smoothing that reduces overfitting, improves
calibration, and stabilizes training by encoding sample confidence and removing
high-frequency noise while preserving causality.

References:
- Triple Barrier with volatility adaptation
- IC-based quality scoring
- Temporal smoothing for financial time series
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal, Union
import numpy as np
import pandas as pd
import warnings


@dataclass
class LabelSmoothingConfig:
    """Configuration for label smoothing pipeline.

    Attributes:
        enabled: Whether to apply label smoothing
        apply_classification_smoothing: Apply epsilon smoothing for classes/probabilities
        apply_uncertainty_shrinkage: Shrink labels based on quality/uncertainty
        apply_causal_ema: Apply temporal smoothing per instrument

        # Pre-processing params (NEW)
        apply_clipping: Clip extreme labels to reduce outliers
        clip_percentile: Percentile for clipping (e.g., 99.0 = clip at 1st and 99th percentiles)
        clip_std_multiplier: Alternatively, clip at mean ± N*std (if percentile=None)
        apply_log_transform: Apply log1p transform to reduce skew (good for trees)
        log_transform_shift: Shift before log transform (default 1.0 for log1p)

        # Classification smoothing params
        epsilon: Label smoothing strength (0.0-0.3). Higher = more smoothing.
                 Default 0.08 means 92% weight on hard class, 8% on uniform.
        temperature: Temperature scaling for probabilities (>1.0 = smoother).
                    Default 1.2. Only used if labels are probabilities.

        # Uncertainty shrinkage params
        gamma: Uncertainty sensitivity (higher = more shrinkage for uncertain samples).
               Default 1.0.
        min_alpha: Minimum weight on label (rest goes to baseline). Prevents over-shrinking.
                   Default 0.12 means at least 12% weight on label.
        baseline: Value to shrink towards. 0.0 for returns, 0.5 for probabilities.
        uncertainty_source: Which metric to use for uncertainty:
            - 'quality_score': Use quality_score from labeler (IC-based)
            - 'quality_inverse': Use 1 - quality_score
            - 'volatility': Use price volatility
            - 'custom': Expect sigma to be passed explicitly

        # Causal EMA params
        ema_decay: Exponential decay factor (0.0-1.0). Higher = more history.
                   0.95 for 15m bars, 0.98 for hourly/daily, 0.9 for faster reaction.
        ema_group_by: Column name to group by for EMA (e.g., 'instrument', 'regime', 'sector')
        ema_seed_method: How to initialize EMA:
            - 'first': Use first value per group
            - 'mean': Use group mean
            - 'zero': Start at 0 (or baseline)

        # Ablation testing
        ablation_mode: For testing individual components:
            - 'full': All enabled components (default)
            - 'classification_only': Only classification smoothing
            - 'uncertainty_only': Only uncertainty shrinkage
            - 'ema_only': Only causal EMA
            - 'classification+uncertainty': First two stages
            - 'uncertainty+ema': Last two stages

        # Monitoring
        store_intermediate: Store intermediate results for debugging
        validate_causality: Check that EMA is truly causal (no lookahead)
    """
    enabled: bool = True

    # Component toggles
    apply_classification_smoothing: bool = True
    apply_uncertainty_shrinkage: bool = True
    apply_causal_ema: bool = True

    # Pre-processing (NEW)
    apply_clipping: bool = True
    clip_percentile: Optional[float] = 99.0  # Clip at 1st and 99th percentiles
    clip_std_multiplier: Optional[float] = None  # Or use mean ± N*std (if percentile=None)
    apply_log_transform: bool = True
    log_transform_shift: float = 1.0  # For log1p

    # Classification smoothing
    epsilon: float = 0.08  # 0.05-0.15 range
    temperature: float = 1.2  # >1.0 for smoother probs

    # Uncertainty shrinkage
    gamma: float = 1.0
    min_alpha: float = 0.12
    baseline: float = 0.0  # or 0.5 for probs
    uncertainty_source: Literal['quality_score', 'quality_inverse', 'volatility', 'custom'] = 'quality_inverse'

    # Causal EMA
    ema_decay: float = 0.95  # 0.9-0.98 typical range
    ema_group_by: Optional[str] = None  # e.g., 'instrument'
    ema_seed_method: Literal['first', 'mean', 'zero'] = 'first'

    # Ablation testing
    ablation_mode: Literal[
        'full',
        'classification_only',
        'uncertainty_only',
        'ema_only',
        'classification+uncertainty',
        'uncertainty+ema'
    ] = 'full'

    # Monitoring
    store_intermediate: bool = False
    validate_causality: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.epsilon < 0 or self.epsilon > 0.3:
            raise ValueError(f"epsilon must be in [0, 0.3], got {self.epsilon}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {self.gamma}")
        if self.min_alpha < 0 or self.min_alpha > 1:
            raise ValueError(f"min_alpha must be in [0, 1], got {self.min_alpha}")
        if self.ema_decay < 0 or self.ema_decay > 1:
            raise ValueError(f"ema_decay must be in [0, 1], got {self.ema_decay}")

        # Apply ablation mode overrides
        self._apply_ablation_mode()

    def _apply_ablation_mode(self):
        """Override component toggles based on ablation_mode."""
        if self.ablation_mode == 'full':
            pass  # All components as configured
        elif self.ablation_mode == 'classification_only':
            self.apply_classification_smoothing = True
            self.apply_uncertainty_shrinkage = False
            self.apply_causal_ema = False
        elif self.ablation_mode == 'uncertainty_only':
            self.apply_classification_smoothing = False
            self.apply_uncertainty_shrinkage = True
            self.apply_causal_ema = False
        elif self.ablation_mode == 'ema_only':
            self.apply_classification_smoothing = False
            self.apply_uncertainty_shrinkage = False
            self.apply_causal_ema = True
        elif self.ablation_mode == 'classification+uncertainty':
            self.apply_classification_smoothing = True
            self.apply_uncertainty_shrinkage = True
            self.apply_causal_ema = False
        elif self.ablation_mode == 'uncertainty+ema':
            self.apply_classification_smoothing = False
            self.apply_uncertainty_shrinkage = True
            self.apply_causal_ema = True


class LabelSmoother:
    """
    Three-stage label smoothing pipeline for robust ML targets.

    Stage 1: Classification/Probability smoothing
        - Softens hard labels to prevent overconfidence
        - Temperature scales probabilities for calibration

    Stage 2: Uncertainty-weighted shrinkage
        - Shrinks uncertain labels towards baseline (0 or 0.5)
        - Uses quality scores, volatility, or custom uncertainty

    Stage 3: Causal EMA per group
        - Removes high-frequency noise via exponential smoothing
        - Maintains causality (no lookahead)
        - Applied per-instrument or per-regime

    Usage:
        config = LabelSmoothingConfig(epsilon=0.08, gamma=1.0, ema_decay=0.95)
        smoother = LabelSmoother(config)

        result = smoother.smooth(
            labels=raw_labels,
            quality_scores=opportunity_quality,
            group_by_data=df[['instrument', 'timestamp']]
        )

        smoothed_labels = result['labels_final']
    """

    def __init__(self, config: LabelSmoothingConfig):
        """Initialize smoother with configuration.

        Args:
            config: Label smoothing configuration
        """
        self.config = config
        self._intermediate = {}  # Store intermediate results if enabled

    def smooth(
        self,
        labels: Union[pd.Series, pd.DataFrame],
        quality_scores: Optional[pd.Series] = None,
        volatility: Optional[pd.Series] = None,
        custom_uncertainty: Optional[pd.Series] = None,
        group_by_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Apply label smoothing pipeline.

        Args:
            labels: Raw labels (Series for single target, DataFrame for multi-target)
            quality_scores: Per-sample quality scores (higher = better)
            volatility: Per-sample volatility (for uncertainty)
            custom_uncertainty: Custom uncertainty metric (higher = more uncertain)
            group_by_data: DataFrame with columns for grouping EMA (e.g., ['instrument', 'timestamp'])
                          Must be aligned with labels index. If ema_group_by is set, this is required.

        Returns:
            Dictionary with:
                - 'labels_final': Final smoothed labels
                - 'labels_raw': Original labels (for reference)
                - 'labels_preprocessing': After clipping/log transform (if applied)
                - 'labels_stage1': After classification smoothing (if applied)
                - 'labels_stage2': After uncertainty shrinkage (if applied)
                - 'labels_stage3': After causal EMA (if applied)
                - 'metadata': Smoothing metadata
        """
        if not self.config.enabled:
            return {'labels_final': labels, 'labels_raw': labels, 'metadata': {'enabled': False}}

        # Initialize result dictionary
        result = {'labels_raw': labels.copy()}
        current_labels = labels.copy()

        # Pre-processing: Clipping and log transform (NEW)
        if self.config.apply_clipping or self.config.apply_log_transform:
            current_labels = self._apply_preprocessing(current_labels)
            result['labels_preprocessing'] = current_labels.copy()

        # Get uncertainty metric if needed for stage 2
        sigma = None
        if self.config.apply_uncertainty_shrinkage:
            sigma = self._get_uncertainty(
                quality_scores=quality_scores,
                volatility=volatility,
                custom_uncertainty=custom_uncertainty,
                labels_index=labels.index
            )

        # Stage 1: Classification/Probability smoothing
        if self.config.apply_classification_smoothing:
            current_labels = self._classification_smooth(current_labels)
            result['labels_stage1'] = current_labels.copy()

        # Stage 2: Uncertainty-weighted shrinkage
        if self.config.apply_uncertainty_shrinkage:
            if sigma is None:
                warnings.warn("Uncertainty shrinkage enabled but no uncertainty metric available. Skipping.")
            else:
                current_labels = self._uncertainty_shrink(current_labels, sigma)
                result['labels_stage2'] = current_labels.copy()

        # Stage 3: Causal EMA per group
        if self.config.apply_causal_ema:
            if self.config.ema_group_by and group_by_data is None:
                warnings.warn("EMA grouping enabled but no group_by_data provided. Skipping EMA.")
            else:
                current_labels = self._causal_ema(current_labels, group_by_data)
                result['labels_stage3'] = current_labels.copy()

        # Final labels
        result['labels_final'] = current_labels

        # Metadata
        result['metadata'] = self._generate_metadata(result)

        # Store intermediate if enabled
        if self.config.store_intermediate:
            self._intermediate = result.copy()

        return result

    def _apply_preprocessing(
        self,
        labels: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply pre-processing: clipping and log transform.

        Args:
            labels: Raw labels

        Returns:
            Pre-processed labels
        """
        labels_processed = labels.copy()

        # Step 1: Clipping extreme values
        if self.config.apply_clipping:
            if self.config.clip_percentile is not None:
                # Percentile-based clipping
                lower_percentile = (100 - self.config.clip_percentile) / 2
                upper_percentile = 100 - lower_percentile

                if isinstance(labels_processed, pd.Series):
                    lower = labels_processed.quantile(lower_percentile / 100)
                    upper = labels_processed.quantile(upper_percentile / 100)
                    labels_processed = labels_processed.clip(lower=lower, upper=upper)
                else:
                    # DataFrame - clip each column
                    for col in labels_processed.columns:
                        lower = labels_processed[col].quantile(lower_percentile / 100)
                        upper = labels_processed[col].quantile(upper_percentile / 100)
                        labels_processed[col] = labels_processed[col].clip(lower=lower, upper=upper)

            elif self.config.clip_std_multiplier is not None:
                # Std-based clipping (mean ± N*std)
                if isinstance(labels_processed, pd.Series):
                    mean = labels_processed.mean()
                    std = labels_processed.std()
                    lower = mean - self.config.clip_std_multiplier * std
                    upper = mean + self.config.clip_std_multiplier * std
                    labels_processed = labels_processed.clip(lower=lower, upper=upper)
                else:
                    # DataFrame - clip each column
                    for col in labels_processed.columns:
                        mean = labels_processed[col].mean()
                        std = labels_processed[col].std()
                        lower = mean - self.config.clip_std_multiplier * std
                        upper = mean + self.config.clip_std_multiplier * std
                        labels_processed[col] = labels_processed[col].clip(lower=lower, upper=upper)

        # Step 2: Log transform to reduce skew
        if self.config.apply_log_transform:
            shift = self.config.log_transform_shift

            if isinstance(labels_processed, pd.Series):
                # Check if labels are already mostly positive or negative
                if labels_processed.min() >= 0:
                    # All positive - apply log1p directly
                    labels_processed = np.sign(labels_processed) * np.log1p(np.abs(labels_processed))
                else:
                    # Mixed signs - apply sign-preserving log transform
                    # log1p(|x|) * sign(x)
                    labels_processed = np.sign(labels_processed) * np.log1p(np.abs(labels_processed) + shift)
            else:
                # DataFrame - transform each column
                for col in labels_processed.columns:
                    if labels_processed[col].min() >= 0:
                        labels_processed[col] = np.sign(labels_processed[col]) * np.log1p(np.abs(labels_processed[col]))
                    else:
                        labels_processed[col] = np.sign(labels_processed[col]) * np.log1p(np.abs(labels_processed[col]) + shift)

        return labels_processed

    def _classification_smooth(
        self,
        labels: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply classification/probability smoothing.

        For binary labels/probabilities:
            p_smooth = (1 - ε) * p + ε * 0.5

        For multiclass (K classes):
            p_smooth = (1 - ε) * p + ε / K

        For continuous labels in [-1, 1]:
            Treat as pseudo-probability and apply smoothing

        Args:
            labels: Raw labels

        Returns:
            Smoothed labels
        """
        eps = self.config.epsilon
        T = self.config.temperature

        if isinstance(labels, pd.Series):
            # Single target - check if binary/continuous
            unique_vals = labels.dropna().unique()

            # If binary (0/1 or -1/1), apply binary smoothing
            if len(unique_vals) <= 2 and set(unique_vals).issubset({-1, 0, 1}):
                # Binary classification smoothing
                # Map to [0, 1] if needed
                if set(unique_vals).issubset({-1, 1}):
                    p = (labels + 1) / 2  # Map [-1, 1] -> [0, 1]
                    p_smooth = (1 - eps) * p + eps * 0.5
                    labels_smooth = 2 * p_smooth - 1  # Map back to [-1, 1]
                else:
                    labels_smooth = (1 - eps) * labels + eps * 0.5

            # If continuous, apply smoothing if in range-like format
            elif labels.min() >= -1.0 and labels.max() <= 1.0:
                # Continuous in [-1, 1], shrink towards 0
                labels_smooth = (1 - eps) * labels

            # If looks like probabilities [0, 1]
            elif labels.min() >= 0.0 and labels.max() <= 1.0:
                # Apply temperature scaling
                if T != 1.0:
                    labels_smooth = self._temperature_scale(labels, T)
                else:
                    labels_smooth = (1 - eps) * labels + eps * 0.5
            else:
                # Other continuous, just apply shrinkage
                labels_smooth = (1 - eps) * labels

            return pd.Series(labels_smooth, index=labels.index, name=labels.name)

        else:
            # DataFrame - multiclass
            K = labels.shape[1]
            labels_array = labels.values
            labels_smooth = (1 - eps) * labels_array + eps / K
            return pd.DataFrame(labels_smooth, index=labels.index, columns=labels.columns)

    def _temperature_scale(self, prob_series: pd.Series, T: float) -> pd.Series:
        """Apply temperature scaling to probabilities.

        Args:
            prob_series: Series of probabilities in [0, 1]
            T: Temperature (>1 = smoother, <1 = sharper)

        Returns:
            Scaled probabilities
        """
        p = prob_series.clip(1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        scaled = 1 / (1 + np.exp(-logit / T))
        return pd.Series(scaled, index=prob_series.index, name=prob_series.name)

    def _uncertainty_shrink(
        self,
        labels: Union[pd.Series, pd.DataFrame],
        sigma: pd.Series
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply uncertainty-weighted shrinkage.

        alpha = 1 / (1 + gamma * sigma)
        alpha = max(alpha, min_alpha)
        label_shrunk = alpha * label + (1 - alpha) * baseline

        Args:
            labels: Labels to shrink
            sigma: Uncertainty metric (higher = more uncertain, more shrinkage)

        Returns:
            Shrunk labels
        """
        gamma = self.config.gamma
        min_alpha = self.config.min_alpha
        baseline = self.config.baseline

        # Calculate alpha (confidence weight)
        alpha = 1.0 / (1.0 + gamma * sigma)
        alpha = np.maximum(alpha, min_alpha)

        # Align alpha with labels
        alpha = alpha.reindex(labels.index, fill_value=min_alpha)

        # Apply shrinkage
        if isinstance(labels, pd.Series):
            labels_shrunk = alpha * labels + (1 - alpha) * baseline
            return pd.Series(labels_shrunk, index=labels.index, name=labels.name)
        else:
            # DataFrame - apply to each column
            labels_shrunk = labels.copy()
            for col in labels.columns:
                labels_shrunk[col] = alpha * labels[col] + (1 - alpha) * baseline
            return labels_shrunk

    def _causal_ema(
        self,
        labels: Union[pd.Series, pd.DataFrame],
        group_by_data: Optional[pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply causal exponential moving average per group.

        EMA formula:
            EMA[i] = decay * EMA[i-1] + (1 - decay) * value[i]

        Args:
            labels: Labels to smooth
            group_by_data: DataFrame with grouping columns (e.g., ['instrument', 'timestamp'])
                          Must be sorted by time within groups!

        Returns:
            EMA-smoothed labels
        """
        decay = self.config.ema_decay

        if isinstance(labels, pd.Series):
            return self._causal_ema_series(labels, group_by_data)
        else:
            # DataFrame - apply to each column
            labels_ema = labels.copy()
            for col in labels.columns:
                labels_ema[col] = self._causal_ema_series(labels[col], group_by_data)
            return labels_ema

    def _causal_ema_series(
        self,
        series: pd.Series,
        group_by_data: Optional[pd.DataFrame]
    ) -> pd.Series:
        """Apply causal EMA to a single series.

        Args:
            series: Series to smooth
            group_by_data: Grouping data

        Returns:
            EMA-smoothed series
        """
        decay = self.config.ema_decay

        if group_by_data is None or self.config.ema_group_by is None:
            # No grouping - single EMA across all data
            ema = self._compute_ema(series.values, decay, self.config.ema_seed_method)
            return pd.Series(ema, index=series.index, name=series.name)

        # With grouping
        group_col = self.config.ema_group_by
        if group_col not in group_by_data.columns:
            raise ValueError(f"Group column '{group_col}' not found in group_by_data")

        # Ensure alignment
        group_by_data = group_by_data.reindex(series.index)

        # Create temporary DataFrame for groupby
        temp_df = pd.DataFrame({
            'label': series,
            'group': group_by_data[group_col]
        })

        # Apply EMA per group
        ema_results = []
        for group_name, group_df in temp_df.groupby('group', sort=False):
            vals = group_df['label'].values
            ema = self._compute_ema(vals, decay, self.config.ema_seed_method)
            ema_series = pd.Series(ema, index=group_df.index)
            ema_results.append(ema_series)

        # Concatenate and sort by original index
        result = pd.concat(ema_results).sort_index()
        result.name = series.name

        return result

    def _compute_ema(
        self,
        values: np.ndarray,
        decay: float,
        seed_method: str
    ) -> np.ndarray:
        """Compute causal EMA for array of values.

        Args:
            values: Array of values
            decay: Decay factor
            seed_method: 'first', 'mean', or 'zero'

        Returns:
            EMA array
        """
        if len(values) == 0:
            return values

        ema = np.empty_like(values, dtype=float)

        # Initialize
        if seed_method == 'first':
            prev = values[0]
        elif seed_method == 'mean':
            prev = np.nanmean(values)
        else:  # 'zero'
            prev = 0.0

        # Compute EMA
        for i, v in enumerate(values):
            if np.isnan(v):
                # Propagate previous EMA for NaN
                ema[i] = prev
            else:
                prev = decay * prev + (1 - decay) * v
                ema[i] = prev

        return ema

    def _get_uncertainty(
        self,
        quality_scores: Optional[pd.Series],
        volatility: Optional[pd.Series],
        custom_uncertainty: Optional[pd.Series],
        labels_index: pd.Index
    ) -> Optional[pd.Series]:
        """Get uncertainty metric based on config.

        Args:
            quality_scores: Quality scores (higher = better)
            volatility: Volatility metric
            custom_uncertainty: Custom uncertainty
            labels_index: Index to align to

        Returns:
            Uncertainty series (higher = more uncertain) or None
        """
        source = self.config.uncertainty_source

        if source == 'custom':
            if custom_uncertainty is None:
                warnings.warn("uncertainty_source='custom' but custom_uncertainty not provided")
                return None
            return custom_uncertainty.reindex(labels_index)

        elif source == 'quality_score':
            if quality_scores is None:
                warnings.warn("uncertainty_source='quality_score' but quality_scores not provided")
                return None
            # Lower quality = higher uncertainty
            # Normalize to [0, 1] range first
            q_norm = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-8)
            uncertainty = 1.0 - q_norm
            return uncertainty.reindex(labels_index)

        elif source == 'quality_inverse':
            if quality_scores is None:
                warnings.warn("uncertainty_source='quality_inverse' but quality_scores not provided")
                return None
            # Direct inverse mapping - assumes quality_scores in [0, 1]
            uncertainty = 1.0 - quality_scores.clip(0, 1)
            return uncertainty.reindex(labels_index)

        elif source == 'volatility':
            if volatility is None:
                warnings.warn("uncertainty_source='volatility' but volatility not provided")
                return None
            # Normalize volatility to reasonable range
            vol_norm = volatility / volatility.median()
            vol_norm = vol_norm.clip(0, 3)  # Cap at 3x median
            return vol_norm.reindex(labels_index)

        else:
            raise ValueError(f"Unknown uncertainty_source: {source}")

    def _generate_metadata(self, result: Dict) -> Dict:
        """Generate metadata about smoothing applied.

        Args:
            result: Result dictionary with intermediate stages

        Returns:
            Metadata dictionary
        """
        metadata = {
            'enabled': True,
            'config': {
                'epsilon': self.config.epsilon,
                'temperature': self.config.temperature,
                'gamma': self.config.gamma,
                'min_alpha': self.config.min_alpha,
                'baseline': self.config.baseline,
                'ema_decay': self.config.ema_decay,
                'ema_group_by': self.config.ema_group_by,
                'ablation_mode': self.config.ablation_mode,
            },
            'stages_applied': {
                'classification_smoothing': self.config.apply_classification_smoothing,
                'uncertainty_shrinkage': self.config.apply_uncertainty_shrinkage,
                'causal_ema': self.config.apply_causal_ema,
            }
        }

        # Compute statistics on label changes
        raw = result['labels_raw']
        final = result['labels_final']

        if isinstance(raw, pd.Series):
            raw_vals = raw.values
            final_vals = final.values
        else:
            raw_vals = raw.values.flatten()
            final_vals = final.values.flatten()

        # Remove NaN
        valid_mask = ~(np.isnan(raw_vals) | np.isnan(final_vals))
        raw_vals = raw_vals[valid_mask]
        final_vals = final_vals[valid_mask]

        if len(raw_vals) > 0:
            diff = final_vals - raw_vals
            metadata['statistics'] = {
                'raw_mean': float(np.mean(raw_vals)),
                'raw_std': float(np.std(raw_vals)),
                'final_mean': float(np.mean(final_vals)),
                'final_std': float(np.std(final_vals)),
                'mean_absolute_change': float(np.mean(np.abs(diff))),
                'max_absolute_change': float(np.max(np.abs(diff))),
                'correlation_raw_final': float(np.corrcoef(raw_vals, final_vals)[0, 1]) if len(raw_vals) > 1 else 1.0,
                'pct_changed': float(np.mean(np.abs(diff) > 1e-6) * 100),  # % of labels that changed
            }

        return metadata

    def get_intermediate_results(self) -> Dict:
        """Get stored intermediate results (if store_intermediate=True).

        Returns:
            Dictionary with intermediate stages
        """
        return self._intermediate.copy()


# Utility functions for ablation testing

def run_ablation_test(
    labels: Union[pd.Series, pd.DataFrame],
    quality_scores: Optional[pd.Series] = None,
    volatility: Optional[pd.Series] = None,
    group_by_data: Optional[pd.DataFrame] = None,
    base_config: Optional[LabelSmoothingConfig] = None,
) -> Dict[str, Dict]:
    """Run ablation test across all smoothing components.

    Args:
        labels: Raw labels
        quality_scores: Quality scores for uncertainty
        volatility: Volatility for uncertainty
        group_by_data: Grouping data for EMA
        base_config: Base configuration (will be copied for each ablation)

    Returns:
        Dictionary mapping ablation_mode -> smoothing result
    """
    if base_config is None:
        base_config = LabelSmoothingConfig()

    ablation_modes = [
        'full',
        'classification_only',
        'uncertainty_only',
        'ema_only',
        'classification+uncertainty',
        'uncertainty+ema'
    ]

    results = {}
    for mode in ablation_modes:
        config = LabelSmoothingConfig(
            enabled=True,
            epsilon=base_config.epsilon,
            temperature=base_config.temperature,
            gamma=base_config.gamma,
            min_alpha=base_config.min_alpha,
            baseline=base_config.baseline,
            ema_decay=base_config.ema_decay,
            ema_group_by=base_config.ema_group_by,
            ema_seed_method=base_config.ema_seed_method,
            uncertainty_source=base_config.uncertainty_source,
            ablation_mode=mode,
            store_intermediate=True
        )

        smoother = LabelSmoother(config)
        result = smoother.smooth(
            labels=labels,
            quality_scores=quality_scores,
            volatility=volatility,
            group_by_data=group_by_data
        )

        results[mode] = result

    # Add no-smoothing baseline
    results['baseline'] = {
        'labels_final': labels.copy(),
        'labels_raw': labels.copy(),
        'metadata': {'enabled': False}
    }

    return results


def compare_ablation_results(
    ablation_results: Dict[str, Dict],
    future_returns: Optional[pd.Series] = None,
    print_summary: bool = True
) -> pd.DataFrame:
    """Compare ablation test results.

    Args:
        ablation_results: Results from run_ablation_test()
        future_returns: Actual future returns for IC calculation
        print_summary: Whether to print summary

    Returns:
        DataFrame with comparison metrics
    """
    metrics = []

    for mode, result in ablation_results.items():
        labels_final = result['labels_final']
        labels_raw = result['labels_raw']

        if isinstance(labels_final, pd.DataFrame):
            # Multi-target - use first column
            labels_final = labels_final.iloc[:, 0]
            labels_raw = labels_raw.iloc[:, 0]

        row = {'mode': mode}

        # Basic statistics
        row['mean'] = labels_final.mean()
        row['std'] = labels_final.std()
        row['skew'] = labels_final.skew()
        row['kurtosis'] = labels_final.kurtosis()

        # Change from raw
        if mode != 'baseline':
            diff = (labels_final - labels_raw).abs()
            row['mean_abs_change'] = diff.mean()
            row['max_abs_change'] = diff.max()
            row['pct_changed'] = (diff > 1e-6).mean() * 100
        else:
            row['mean_abs_change'] = 0
            row['max_abs_change'] = 0
            row['pct_changed'] = 0

        # Information Coefficient if future returns provided
        if future_returns is not None:
            aligned_labels = labels_final.reindex(future_returns.index)
            valid_mask = ~(aligned_labels.isna() | future_returns.isna())
            if valid_mask.sum() > 10:
                ic = aligned_labels[valid_mask].corr(future_returns[valid_mask], method='spearman')
                row['IC'] = ic
            else:
                row['IC'] = np.nan

        metrics.append(row)

    df = pd.DataFrame(metrics)

    if print_summary:
        print("\n" + "="*80)
        print("LABEL SMOOTHING ABLATION TEST RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

    return df
