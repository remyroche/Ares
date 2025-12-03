"""Centralized constants for meta-label outputs and diagnostics.

These constants are used across pre-training steps to:
- Exclude label-like columns from feature matrices
- Identify primary targets for optimization / model training
- Keep diagnostic columns (e.g. R-multiple, TTO) available for analysis

DIRECTIONAL LABELS (2025 addition):
- binary_label_long: Binary success/failure for LONG trades only (NaN for shorts)
- binary_label_short: Binary success/failure for SHORT trades only (NaN for longs)

These directional labels allow training separate classifiers for each direction,
avoiding the issue where a unified binary_label trains equally on longs and shorts,
leading to near-zero edge on both sides.

For classification tasks, prefer:
- binary_label_long when training a long-only classifier
- binary_label_short when training a short-only classifier

For regression tasks, use:
- target_long for long-only regression
- target_short for short-only regression
"""

META_LABEL_TARGET_COLUMNS = [
    'binary_label',
    'binary_label_long',   # NEW: Directional binary label for longs only
    'binary_label_short',  # NEW: Directional binary label for shorts only
    'smoothed_label',
    'realized_return',
    'label_uncertainty',
    'target_long_fused',
    'target_short_fused',
    'target_long',
    'target_short',
    # Fused directional aliases emitted by some labeling steps
    'fused_target_long',
    'fused_target_short',
    'target',
    'label',
    'return',
    'price_target_vol_normalized',
]

# Primary training targets organized by use case:
# - CLASSIFICATION: binary_label_long, binary_label_short (direction-specific)
# - REGRESSION: target_long, target_short (expected returns)
# - LEGACY: binary_label (unified, for backward compatibility)
META_LABEL_PRIMARY_TRAINING_TARGETS = [
    'binary_label',
    'binary_label_long',   # NEW: Preferred for long-only classifiers
    'binary_label_short',  # NEW: Preferred for short-only classifiers
    'target_long_fused',
    'target_short_fused',
    'target_long',
    'target_short',
    'fused_target_long',
    'fused_target_short',
    'target',
    'label',
    'return',
    'price_target_vol_normalized',
]

# Mapping from training direction to appropriate classification target
# NOTE: 'both' direction uses binary_label_long as primary (no legacy binary_label)
DIRECTIONAL_CLASSIFICATION_TARGETS = {
    'long': 'binary_label_long',
    'short': 'binary_label_short',
    'both': 'binary_label_long',  # For 'both', prefer long as primary signal
}

# Mapping from training direction to appropriate regression target
DIRECTIONAL_REGRESSION_TARGETS = {
    'long': 'target_long',
    'short': 'target_short',
}

META_LABEL_DIAGNOSTIC_COLUMNS = [
    # Meta-model outputs / weights
    'meta_probability',
    'target_sample_weight',
    'labeling_method_id',
    'labeling_timestamp',
    # Event-level diagnostics
    'r_multiple',
    'event_duration_bars',
    'exit_reason',
    'event_tto_mean_last_50',
    'event_r_multiple_mean_last_50',
]

# Convenience unions used by feature-generation steps
META_LABEL_EXCLUDED_FEATURE_COLUMNS = (
    META_LABEL_TARGET_COLUMNS
    + META_LABEL_DIAGNOSTIC_COLUMNS
)
