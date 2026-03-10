import re

with open('extreme_price_movements/features.py', 'r') as f:
    content = f.read()

# We need to insert calls to add_cross_sectional_peer_context_features and add_time_series_percentile_features
# just before the CausalFeatureTransformer pass.

insertion_point_str = '''    if requested_feature_set:
        feats = {k: v for k, v in feats.items() if k in requested_feature_set}
    tprint(f"Features: {len(feats)} features before CausalTransform. Applying transforms...")'''

new_str = '''    # --- explicit peer-context and ts-percentile features ---
    cs_feats = add_cross_sectional_peer_context_features(feats, min_group_size=5)
    feats.update(cs_feats)

    ts_feats = add_time_series_percentile_features(feats, lookback=720, min_history_fraction=0.25)
    feats.update(ts_feats)

    if requested_feature_set:
        feats = {k: v for k, v in feats.items() if k in requested_feature_set}
    tprint(f"Features: {len(feats)} features before CausalTransform. Applying transforms...")'''

content = content.replace(insertion_point_str, new_str)

with open('extreme_price_movements/features.py', 'w') as f:
    f.write(content)
