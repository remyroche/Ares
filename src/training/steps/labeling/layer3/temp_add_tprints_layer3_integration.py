import re

# Read the file
with open('layer3/core.py', 'r') as f:
    content = f.read()

# Enhance the streamlined Layer 3 features integration
streamlined_pattern = r'(        # Add Streamlined Layer 3 Causal Features \(8 essential features\)\s+layer3_streamlined_enabled = cfg\.get\("layer3_streamlined_enabled", True\)\s+if layer3_streamlined_enabled and len\(meta_features\) > 5:.*?else:\s+tprint_info\("⏭️ Skipping streamlined Layer 3 features \(disabled or insufficient features\)"\))'

streamlined_replacement = r'''        # Add Streamlined Layer 3 Causal Features (8 essential features)
        layer3_streamlined_enabled = cfg.get("layer3_streamlined_enabled", True)
        if layer3_streamlined_enabled and len(meta_features) > 5:
            tprint_info(">>> Layer 3: Adding Streamlined Causal Meta-Features (8 features)...")
            try:
                from ..streamlined_layer3_features import StreamlinedLayer3Features, quick_streamlined_layer3_features

                # Initialize streamlined feature generator
                tprint_info("   🔧 Initializing streamlined feature generator...")
                streamlined_generator = StreamlinedLayer3Features(
                    surprise_threshold=2.0,
                    rolling_window=20,
                    n_clusters=3,
                    verbose=True
                )
                tprint_success("   ✅ Streamlined generator initialized")

                # Generate streamlined features
                tprint_info("   📊 Generating streamlined features...")
                df_subset = df.loc[df.index.intersection(meta_features)]
                if len(df_subset) > 0:
                    tprint_info(f"      📝 Using subset: {len(df_subset)} samples, {len(meta_features)} features")
                    
                    # Create custom features for streamlined generation
                    tprint_info("   🔧 Creating custom features for streamlined generation...")
                    custom_features = pd.DataFrame(index=df_subset.index)
                    if 'volatility_1d' in df.columns:
                        custom_features['volatility'] = df.loc[df_subset.index, 'volatility_1d']
                        tprint_info("      ✅ Added volatility feature")
                    else:
                        tprint_warning("      ⚠️ No volatility_1d column found")
                    
                    # Generate streamlined features (8 essential features)
                    tprint_info("   🎯 Generating 8 streamlined causal features...")
                    streamlined_features = streamlined_generator.generate_streamlined_features(
                        df=df_subset,
                        base_model_cols=safe_base_cols,
                        target_col=target_col,
                        custom_features=custom_features,
                        specialist_predictions=None,  # Would use actual specialist predictions
                        causal_effects=None  # Would use actual causal effects
                    )

                    # Add streamlined features to meta-features
                    tprint_info("   🔗 Adding streamlined features to meta-features...")
                    streamlined_cols_added = []
                    for col in streamlined_features.columns:
                        feature_name = f"streamlined_{col}"
                        if feature_name not in meta_features:
                            df.loc[streamlined_features.index, feature_name] = streamlined_features[col]
                            meta_features.append(feature_name)
                            streamlined_cols_added.append(feature_name)

                    if streamlined_cols_added:
                        tprint_success(f"   ✅ Added {len(streamlined_cols_added)} streamlined causal meta-features")
                        tprint_info(f"      📝 Features: {streamlined_cols_added}")
                        tprint_info(f"   📊 New total: {len(meta_features)} features (+{len(streamlined_cols_added)} streamlined)")
                    else:
                        tprint_warning("   ⚠️ No streamlined features added")
                else:
                    tprint_warning("   ⚠️ Insufficient data for streamlined features")

            except Exception as e:
                tprint_error(f"   ❌ Streamlined Layer 3 features failed: {e}")
                import traceback
                tprint_error(f"   📋 Traceback: {traceback.format_exc()}")
        else:
            if not layer3_streamlined_enabled:
                tprint_info("⏭️ Skipping streamlined Layer 3 features (disabled)")
            else:
                tprint_info("⏭️ Skipping streamlined Layer 3 features (insufficient features)")'''

# Apply the replacement
new_content = re.sub(streamlined_pattern, streamlined_replacement, content, flags=re.DOTALL)

# Write back to file
with open('layer3/core.py', 'w') as f:
    f.write(new_content)

print("Added comprehensive t-prints to streamlined Layer 3 features in layer3/core.py")
