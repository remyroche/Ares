"""Feature View definitions for different model families."""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_feature_view(all_features: List[str], view_name: str) -> List[str]:
    """
    Returns a subset of features belonging to the requested view.

    view_name: 'X_linear' or 'X_tree'
    """
    selected_features = []

    for feat in all_features:
        if view_name == "X_linear":
            # Linear view: Include continuous features, cs_rz_, FFD. Exclude rank and ts_pct duplicates.
            if feat.startswith("cs_rank_") or feat.startswith("ts_pct_"):
                # But allow some specific ones if they have no continuous equivalent?
                # Generally we exclude percentile/rank duplicates in linear
                continue

            # If there's an explicit continuous and we have both, we keep the continuous.
            # cs_rz_ is explicitly for linear
            selected_features.append(feat)

        elif view_name == "X_tree":
            # Tree view: Include continuous, rank, ts_pct. Exclude cs_rz_ by default.
            # De-emphasize FFD (unless it's specifically useful, but we can just exclude some or keep base)
            if feat.startswith("cs_rz_"):
                continue

            # Reduce FFD spam in trees
            if feat.startswith("ffd_") and not feat.startswith("ffd_diff_1_"):
                # Keep some basic FFD like diff 1
                # But maybe exclude heavy FFD channel features?
                pass

            selected_features.append(feat)

        else:
            selected_features.append(feat)

    return selected_features
