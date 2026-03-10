"""Feature View definitions for different model families."""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Requirement 4 & 5: Magnitude features with percentile companions
MAGNITUDE_SENSITIVE_FEATURES = {
    "ret1h", "ret24h", "rv_24h", "atr_pct",
    "dist_vwap_norm", "dist_ema_fast", "gap_pct", "range_pct"
}

def get_feature_metadata(feature_name: str) -> Dict:
    """
    Generates metadata registry for a feature dynamically based on rules.
    This fulfills the requirement for a feature metadata registry with view eligibility flags.
    """
    is_ffd = feature_name.startswith("ffd_")
    is_cs_rank = feature_name.startswith("cs_rank_")
    is_cs_rz = feature_name.startswith("cs_rz_")
    is_ts_pct = feature_name.startswith("ts_pct_")

    # Base signal group extraction
    base_signal = feature_name
    if is_cs_rank:
        base_signal = feature_name.replace("cs_rank_", "")
    elif is_cs_rz:
        base_signal = feature_name.replace("cs_rz_", "")
    elif is_ts_pct:
        base_signal = feature_name.replace("ts_pct_", "")

    canonical_form = "continuous"
    if is_cs_rank or is_ts_pct:
        canonical_form = "percentile_rank"
    elif is_cs_rz:
        canonical_form = "robust_z"

    # View Eligibility
    eligible_for_linear = True
    eligible_for_tree = True

    if is_cs_rank or is_ts_pct:
        eligible_for_linear = False  # De-duplicate: no rank/pct forms for linear

    if is_cs_rz:
        eligible_for_tree = False  # Trees prefer cs_rank_

    if is_ffd:
        # Limit FFD in trees to basic
        if not (feature_name.startswith("ffd_diff_1_") or feature_name.startswith("ffd_strength_")):
            eligible_for_tree = False

    return {
        "feature_name": feature_name,
        "base_signal_group": base_signal,
        "canonical_form": canonical_form,
        "eligible_for_linear": eligible_for_linear,
        "eligible_for_tree": eligible_for_tree,
        "has_percentile_companion": base_signal in MAGNITUDE_SENSITIVE_FEATURES,
        "has_peer_context_variant": base_signal in MAGNITUDE_SENSITIVE_FEATURES, # proxy
        "ffd_eligible": is_ffd, # If it is one, the family is
    }

def get_feature_view(all_features: List[str], view_name: str) -> List[str]:
    """
    Returns a subset of features belonging to the requested view,
    enforcing deduplication and model-specific selection logic
    using the feature metadata registry.

    view_name: 'X_linear' or 'X_tree'
    """
    selected_features = []

    for feat in all_features:
        meta = get_feature_metadata(feat)

        if view_name == "X_linear" and meta["eligible_for_linear"]:
            selected_features.append(feat)

        elif view_name == "X_tree" and meta["eligible_for_tree"]:
            selected_features.append(feat)

    return selected_features
