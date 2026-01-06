import re

# Read the file
with open('layer3/core.py', 'r') as f:
    content = f.read()

# Find the location after hierarchical filtering and add minimal causal features
pattern = r'(        # Hierarchical filtering for performance\n)(        pre_filter_count = len\(meta_features\))'

replacement = r'\1\2\n        \n        # Add Minimal Layer 3 Causal Features (Modern De Prado)\n        layer3_minimal_causal_enabled = cfg.get("layer3_minimal_causal_enabled", True)\n        if layer3_minimal_causal_enabled and len(meta_features) > 10:\n            tprint_info(">>> Layer 3: Adding Minimal Causal Meta-Features...")\n            try:\n                from ..minimal_causal_features import generate_minimal_layer3_features\n                \n                # Generate confounder features\n                from ..ohlcv_regime_features import OHLCVRegimeFeatures\n                regime_generator = OHLCVRegimeFeatures(\n                    volatility_window=20,\n                    trend_window=20,\n                    enable_volatility_regimes=True,\n                    enable_trend_features=True,\n                    enable_microstructure_proxies=True,\n                    verbose=False\n                )\n                \n                df_subset = df.loc[meta_features.index]\n                custom_features = regime_generator.generate_features(df_subset)\n                \n                # Generate minimal causal meta-features (3-4 features only)\n                minimal_causal_features = generate_minimal_layer3_features(\n                    df=meta_features,\n                    base_model_cols=safe_base_cols,\n                    target_col=target_col,\n                    custom_features=custom_features,\n                    surprise_threshold=2.0,\n                    rolling_window=20,\n                    verbose=True\n                )\n                \n                # Add minimal causal features to meta-features\n                causal_cols_added = []\n                for col in minimal_causal_features.columns:\n                    if col not in meta_features.columns:\n                        meta_features[col] = minimal_causal_features[col]\n                        causal_cols_added.append(col)\n                \n                tprint_success(f"   ✅ Added {len(causal_cols_added)} minimal causal meta-features")\n                tprint_info(f"   - Total new features: {len(causal_cols_added)} (maximum 4)")\n                \n                # Store for downstream use\n                self.layer3_minimal_causal_features = minimal_causal_features\n                \n            except Exception as e:\n                tprint_warning(f"   ⚠️ Minimal Layer 3 causal features failed: {e}")\n                self.layer3_minimal_causal_features = pd.DataFrame(index=meta_features.index)\n        else:\n            tprint_info("⏭️ Skipping minimal Layer 3 causal features (disabled or insufficient data)")\n            self.layer3_minimal_causal_features = pd.DataFrame(index=meta_features.index)\n        \n        # Update meta-features count after minimal causal features\n        post_causal_count = len(meta_features)'

# Apply the replacement
new_content = re.sub(pattern, replacement, content)

# Write back to file
with open('layer3/core.py', 'w') as f:
    f.write(new_content)

print("Added minimal Layer 3 causal features integration")
