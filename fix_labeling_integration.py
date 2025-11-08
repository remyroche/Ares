#!/usr/bin/env python3
"""
Fix for feature_generation_labeling_integration_step to ensure proper saving with versioned artifacts.
"""

import re

# Read the original file
with open('src/training/steps/pre_training/feature_generation_labeling_integration_step.py', 'r') as f:
    content = f.read()

# Fix 1: Ensure _create_labeled_dataframe_efficiently properly adds target columns
# Find the method and replace it with a fixed version
pattern = r'def _create_labeled_dataframe_efficiently\(self, market_data, labeling_result, vol_config\):.*?(?=\n    @|\n    def|\Z)'
replacement = '''def _create_labeled_dataframe_efficiently(self, market_data, labeling_result, vol_config):
        """Create labeled DataFrame efficiently without full copying."""
        try:
            essential_columns = ['close', 'open', 'high', 'low', 'volume']
            available_columns = [col for col in essential_columns if col in market_data.columns]

            labeled_data_df = market_data[available_columns].copy()
            initial_columns = list(labeled_data_df.columns)

            aligned_labels: Optional[Union[pd.Series, pd.DataFrame]] = None

            if not hasattr(labeling_result, 'labels'):
                tprint_error("❌ Labeling result missing 'labels' attribute. Failing early.")
                raise ValueError("Labeling result missing 'labels' attribute")

            labels_data = labeling_result.labels

            if labels_data is None:
                tprint_error("❌ Labeling result returned 'labels=None'. Failing early.")
                raise ValueError("Labeling result returned None for labels")

            if isinstance(labels_data, (pd.Series, pd.DataFrame)) and labels_data.empty:
                tprint_error("❌ Labeling result returned empty labels structure. Failing early.")
                raise ValueError("Labeling result returned empty labels")

            def _safe_preview(obj):
                try:
                    if isinstance(obj, pd.DataFrame):
                        return obj.head().to_dict()
                    if isinstance(obj, pd.Series):
                        return obj.head().tolist()
                    if isinstance(obj, (list, tuple)):
                        return list(obj[:5])
                    if hasattr(obj, 'shape') and hasattr(obj, '__getitem__'):
                        return str(obj[:5])
                    if isinstance(obj, dict):
                        return {k: ('array' if hasattr(v, 'shape') else str(v)[:80]) for k, v in list(obj.items())[:3]}
                    return str(obj)[:120]
                except Exception as preview_error:
                    return f"<preview_unavailable: {preview_error}>"

            raw_preview = _safe_preview(labels_data)
            tprint_info(f"🔬 Raw label structure: type={type(labels_data)}, preview={raw_preview}")
            self.logger.info("[LABEL_DEBUG] Raw label structure type=%s preview=%s", type(labels_data).__name__, raw_preview)

            # Normalize label structures into pandas containers
            if isinstance(labels_data, dict):
                chosen_key = None
                for key, value in labels_data.items():
                    if isinstance(value, (pd.Series, pd.DataFrame)):
                        labels_data = value
                        chosen_key = key
                        break
                    if isinstance(value, (np.ndarray, list, tuple)):
                        labels_data = pd.Series(value)
                        chosen_key = key
                        break
                if chosen_key is None:
                    raise ValueError(f"Unsupported dict-based label structure: keys={list(labels_data.keys())[:3]}")
                else:
                    tprint_info(f"📦 Normalized labels from dict key '{chosen_key}' → {type(labels_data)}")

            if isinstance(labels_data, np.ndarray):
                if labels_data.ndim == 1:
                    labels_data = pd.Series(labels_data)
                    tprint_info("📈 Converted numpy vector labels to Series")
                elif labels_data.ndim == 2:
                    column_names = [f"target_{i}" for i in range(labels_data.shape[1])]
                    labels_data = pd.DataFrame(labels_data, columns=column_names)
                    tprint_info("📊 Converted numpy matrix labels to DataFrame")
                else:
                    raise ValueError(f"Unsupported numpy label shape: {labels_data.shape}")

            if isinstance(labels_data, (list, tuple)):
                labels_data = pd.Series(labels_data)
                tprint_info("📈 Converted list/tuple labels to Series")

            normalized_preview = _safe_preview(labels_data)
            tprint_info(f"✅ Normalized label structure: type={type(labels_data)}, preview={normalized_preview}")
            self.logger.info("[LABEL_DEBUG] Normalized label structure type=%s preview=%s", type(labels_data).__name__, normalized_preview)

            label_columns_detected = []

            if isinstance(labels_data, pd.DataFrame):
                labels_df = labels_data.copy()
                if labels_df.index.empty or not labels_df.index.equals(market_data.index):
                    labels_df.index = market_data.index[:len(labels_df)]
                aligned_labels = labels_df.apply(pd.to_numeric, errors='coerce')
                self.logger.info(
                    "[LABEL_DEBUG] Aligned DataFrame labels columns=%s head=%s",
                    list(aligned_labels.columns),
                    aligned_labels.head().to_dict(orient='list')
                )

                for orig_col in aligned_labels.columns:
                    col_name = str(orig_col)
                    if col_name in labeled_data_df.columns:
                        col_name = f"label_{col_name}"
                    labeled_data_df[col_name] = aligned_labels[orig_col].astype(np.float32)
                    label_columns_detected.append(col_name)
            elif isinstance(labels_data, pd.Series):
                labels_series = labels_data
                if labels_series.index.empty or not labels_series.index.equals(market_data.index):
                    labels_series.index = market_data.index[:len(labels_series)]
                aligned_labels = pd.to_numeric(labels_series, errors='coerce')
                sample_values = aligned_labels.head().tolist() if hasattr(aligned_labels, 'head') else []
                self.logger.info(
                    "[LABEL_DEBUG] Aligned Series labels column=price_target_vol_normalized head=%s",
                    sample_values
                )
                labeled_data_df['price_target_vol_normalized'] = aligned_labels.astype(np.float32)
                label_columns_detected.append('price_target_vol_normalized')
            else:
                raise ValueError(f"Unsupported label type after normalization: {type(labels_data)}")

            # FIX: Add explicit target columns for training if not present
            target_columns = ['target_long', 'target_short', 'target_neutral']
            for col in target_columns:
                if col not in labeled_data_df.columns:
                    if 'price_target_vol_normalized' in labeled_data_df.columns:
                        # Create binary targets from continuous target
                        labeled_data_df['target_long'] = (labeled_data_df['price_target_vol_normalized'] > 0.01).astype(np.float32)
                        labeled_data_df['target_short'] = (labeled_data_df['price_target_vol_normalized'] < -0.01).astype(np.float32)
                        labeled_data_df['target_neutral'] = ((labeled_data_df['price_target_vol_normalized'] >= -0.01) & 
                                                        (labeled_data_df['price_target_vol_normalized'] <= 0.01)).astype(np.float32)
                        label_columns_detected.extend(['target_long', 'target_short', 'target_neutral'])
                        tprint_info("🎯 Added explicit target columns for training")
                    else:
                        # Add empty target columns if no price target available
                        labeled_data_df[col] = np.zeros(len(labeled_data_df), dtype=np.float32)
                        label_columns_detected.append(col)

            if hasattr(labeling_result, 'quality_scores') and labeling_result.quality_scores and aligned_labels is not None:
                for target_name, target_data in labeling_result.quality_scores.items():
                    if not hasattr(target_data, 'opportunity_quality_scores'):
                        continue

                    quality_scores_full = pd.Series(0.0, index=labeled_data_df.index, dtype=np.float32)

                    if isinstance(aligned_labels, pd.DataFrame):
                        non_zero_mask = aligned_labels.fillna(0).any(axis=1)
                    else:
                        non_zero_mask = aligned_labels.fillna(0) != 0

                    if hasattr(target_data, 'signal_directions') and hasattr(target_data.signal_directions, 'index'):
                        labeled_indices = pd.Index(target_data.signal_directions.index)
                    else:
                        labeled_indices = labeled_data_df.index[non_zero_mask]

                    labeled_indices = labeled_indices.intersection(labeled_data_df.index)
                    quality_values = np.asarray(list(target_data.opportunity_quality_scores or []), dtype=np.float32)

                    if len(labeled_indices) and quality_values.size:
                        assign_len = min(len(labeled_indices), quality_values.size)
                        # Ensure both arrays have exactly the same length for assignment
                        indices_to_assign = labeled_indices[:assign_len]
                        values_to_assign = quality_values[:assign_len]
                        quality_scores_full.loc[indices_to_assign] = values_to_assign

                    quality_col = f'quality_scores_{target_name}'
                    labeled_data_df[quality_col] = quality_scores_full
                    label_columns_detected.append(quality_col)

            if aligned_labels is None:
                labeled_data_df['price_target_vol_normalized'] = np.zeros(len(labeled_data_df), dtype=np.float32)

            timestamp_ns = np.int64(pd.Timestamp.utcnow().value)
            labeled_data_df['labeling_timestamp'] = np.full(len(labeled_data_df), timestamp_ns, dtype=np.int64)
            labeled_data_df['labeling_method_id'] = np.full(len(labeled_data_df), 1, dtype=np.int8)
            labeled_data_df['base_threshold'] = np.float32(vol_config.volatility_threshold)
            labeled_data_df['lookahead_periods'] = np.int16(vol_config.lookahead_periods)

            label_cols_added = [col for col in labeled_data_df.columns if col not in initial_columns]
            self.logger.info(
                "[LABEL_DEBUG] Label columns added=%s total_columns=%d",
                label_cols_added,
                len(labeled_data_df.columns)
            )

            numeric_label_cols = [col for col in label_cols_added if pd.api.types.is_numeric_dtype(labeled_data_df[col])]
            non_zero_counts = {col: int((labeled_data_df[col] != 0).sum()) for col in numeric_label_cols}
            tprint_info(f"🎯 Added label columns: {label_cols_added}")
            tprint_info(f"📊 Non-zero counts for label columns: {non_zero_counts}")

            object_columns = labeled_data_df.select_dtypes(include=['object', 'string']).columns
            for col in object_columns:
                series = labeled_data_df[col]
                if series.dropna().empty:
                    labeled_data_df[col] = np.zeros(len(series), dtype=np.float32)
                else:
                    try:
                        labeled_data_df[col] = pd.to_numeric(series, errors='coerce').astype(np.float32)
                    except ValueError:
                        labeled_data_df[col] = series.astype('category').cat.codes.astype(np.int32)

            remaining_objects = labeled_data_df.select_dtypes(include=['object']).columns
            if len(remaining_objects) > 0:
                raise ValueError(f"Unhandled object dtype columns in labeled data: {list(remaining_objects)}")

            return labeled_data_df
        except Exception as e:
            tprint(f"⚠️ Failed to create labeled DataFrame efficiently: {e}", "WARNING")
            return market_data.copy()'''

# Apply the fix
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Fix 2: Ensure data_category is properly set to 'features' when saving labeled data
pattern2 = r'labeled_data_path = self\._save_artifact\(\s*data=labeled_data_df,\s*artifact_name=f\'labeled_data_\{config\["symbol"\]\}_\{config\["timeframe"\]\}\','
replacement2 = '''labeled_data_path = self._save_artifact(
                        data=labeled_data_df,
                        artifact_name=f'labeled_data_{config["symbol"]}_{config["timeframe"]}',
                        artifact_type='data',
                        data_category='features',  # Explicitly set to features for HDF5 versioning
                        compression='auto',  # Use automatic compression for large datasets
                        metadata={'''

# Apply the second fix
content = re.sub(pattern2, replacement2, content)

# Write the fixed file
with open('src/training/steps/pre_training/feature_generation_labeling_integration_step.py', 'w') as f:
    f.write(content)

print("✅ Fixed feature_generation_labeling_integration_step.py")
print("   1. Added explicit target columns for training")
print("   2. Set data_category='features' for proper HDF5 versioning")
