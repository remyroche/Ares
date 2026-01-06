import re

# Read the file
with open('layer3/core.py', 'r') as f:
    content = f.read()

# Find the location after minimal causal features and add streamlined features
pattern = r'(        else:\n            tprint_info\("⏭️ Skipping Layer 2\.5 Chaser features \(disabled or unavailable\)"\)\n        \n)(    for pattern, count in pattern_counts\.items\(\):)'

# Add streamlined Layer 3 features
streamlined_features = '''        # Add Streamlined Layer 3 Causal Features (8 essential features)
        layer3_streamlined_enabled = cfg.get("layer3_streamlined_enabled", True)
        if layer3_streamlined_enabled and len(meta_features) > 5:
            tprint_info(">>> Layer 3: Adding Streamlined Causal Meta-Features (8 features)...")
            try:
                from ..streamlined_layer3_features import StreamlinedLayer3Features, quick_streamlined_layer3_features

                # Initialize streamlined feature generator
                streamlined_generator = StreamlinedLayer3Features(
                    surprise_threshold=2.0,
                    rolling_window=20,
                    n_clusters=3,
                    verbose=True
                )

                # Generate streamlined features
                df_subset = df.loc[df.index.intersection(meta_features)]
                if len(df_subset) > 0:
                    # Create custom features for streamlined generation
                    custom_features = pd.DataFrame(index=df_subset.index)
                    if 'volatility_1d' in df.columns:
                        custom_features['volatility'] = df.loc[df_subset.index, 'volatility_1d']
                    
                    # Generate streamlined features (8 essential features)
                    streamlined_features = streamlined_generator.generate_streamlined_features(
                        df=df_subset,
                        base_model_cols=safe_base_cols,
                        target_col=target_col,
                        custom_features=custom_features,
                        specialist_predictions=None,  # Would use actual specialist predictions
                        causal_effects=None  # Would use actual causal effects
                    )

                    # Add streamlined features to meta-features
                    streamlined_cols_added = []
                    for col in streamlined_features.columns:
                        feature_name = f"streamlined_{col}"
                        if feature_name not in meta_features:
                            df.loc[streamlined_features.index, feature_name] = streamlined_features[col]
                            meta_features.append(feature_name)
                            streamlined_cols_added.append(feature_name)

                    if streamlined_cols_added:
                        tprint_success(f"   ✅ Added {len(streamlined_cols_added)} streamlined causal meta-features")
                        tprint_info(f"   - New total: {len(meta_features)} features (+{len(streamlined_cols_added)} streamlined)")
                    else:
                        tprint_warning("   ⚠️ No streamlined features added")
                else:
                    tprint_warning("   ⚠️ Insufficient data for streamlined features")

            except Exception as e:
                tprint_warning(f"   ⚠️ Streamlined Layer 3 features failed: {e}")
        else:
            tprint_info("⏭️ Skipping streamlined Layer 3 features (disabled or insufficient features)")
        
        '''

# Apply the replacement
new_content = re.sub(pattern, r'\1' + streamlined_features + r'\n\2', content)

# Write back to file
with open('layer3/core.py', 'w') as f:
    f.write(new_content)

print("Added streamlined Layer 3 features to layer3/core.py")
