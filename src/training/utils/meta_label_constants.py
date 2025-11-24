"""Centralized constants for meta-label outputs and diagnostics.

These constants are used across pre-training steps to:
- Exclude label-like columns from feature matrices
- Identify primary targets for optimization / model training
- Keep diagnostic columns (e.g. R-multiple, TTO) available for analysis
"""

META_LABEL_TARGET_COLUMNS = [
    'binary_label',
    'smoothed_label',
    'realized_return',
    'label_uncertainty',
    'target_long_fused',
    'target_short_fused',
    'target_long',
    'target_short',
    'target',
    'label',
    'return',
    'price_target_vol_normalized',
]

META_LABEL_PRIMARY_TRAINING_TARGETS = [
    'binary_label',
    'target_long_fused',
    'target_short_fused',
    'target_long',
    'target_short',
    'target',
    'label',
    'return',
    'price_target_vol_normalized',
]

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
