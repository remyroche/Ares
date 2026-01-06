import re

# Read the file
with open('layer3/core.py', 'r') as f:
    content = f.read()

# Find the location after minimal causal features and before geometry generation
pattern = r'(        tprint_info\("⏭️ Skipping minimal Layer 3 causal features \(disabled or insufficient features\)"\n)(        layer3_minimal_causal_features = pd\.DataFrame\(index=df\.index\)\n        \n)(    for pattern, count in pattern_counts\.items\(\):)'

replacement = r'\1\2\3\n        \n        # Add Layer 2.5 Chaser Features (if available)\n        layer25_chaser_enabled = cfg.get("layer25_chaser_enabled", False)\n        if layer25_chaser_enabled and hasattr(self, "layer2_chaser_results"):\n            tprint_info(">>> Layer 3: Adding Layer 2.5 Chaser Features...")\n            try:\n                chaser_results = self.layer2_chaser_results\n                chaser_features_added = []\n                \n                for gt_uuid, chaser_data in chaser_results.items():\n                    if "meta_features" in chaser_data:\n                        meta_features_df = chaser_data["meta_features"]\n                        \n                        # Align indices\n                        common_index = df.index.intersection(meta_features_df.index)\n                        if len(common_index) > 0:\n                            meta_features_aligned = meta_features_df.loc[common_index]\n                            \n                            # Add Chaser features with UUID prefix\n                            for col in meta_features_aligned.columns:\n                                feature_name = f"chaser_{gt_uuid}_{col}"\n                                if feature_name not in meta_features:\n                                    df.loc[common_index, feature_name] = meta_features_aligned[col]\n                                    meta_features.append(feature_name)\n                                    chaser_features_added.append(feature_name)\n                        \n                        # Add conflict intensity as separate feature\n                        if "conflict_intensity" in chaser_data:\n                            conflict_series = pd.Series(\n                                chaser_data["conflict_intensity"],\n                                index=common_index[:len(chaser_data["conflict_intensity"])]\n                            )\n                            conflict_feature = f"chaser_{gt_uuid}_conflict_intensity"\n                            if conflict_feature not in meta_features:\n                                df.loc[conflict_series.index, conflict_feature] = conflict_series\n                                meta_features.append(conflict_feature)\n                                chaser_features_added.append(conflict_feature)\n                \n                if chaser_features_added:\n                    tprint_success(f"   ✅ Added {len(chaser_features_added)} Chaser features")\n                    tprint_info(f"   - Total Chaser features: {len(chaser_features_added)}")\n                    tprint_info(f"   - New meta-feature total: {len(meta_features)}")\n                else:\n                    tprint_warning("   ⚠️ No Chaser features added")\n                \n            except Exception as e:\n                tprint_warning(f"   ⚠️ Chaser feature integration failed: {e}")\n        else:\n            tprint_info("⏭️ Skipping Layer 2.5 Chaser features (disabled or unavailable)")\n        '

# Apply the replacement
new_content = re.sub(pattern, replacement, content)

# Write back to file
with open('layer3/core.py', 'w') as f:
    f.write(new_content)

print("Added Layer 2.5 Chaser integration to layer3/core.py")
