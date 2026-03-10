import re

with open('extreme_price_movements/feature_transforms.py', 'r') as f:
    content = f.read()

# Fix transform_batch as well
content = content.replace(
    '''    def transform_batch(
        self,
        feats: dict,
        skip_keys: set,
        chunk_size: int = 50,
    ) -> dict:
        """
        Transform all features in batched stacked-matrix calls.

        * Features in *skip_keys* are cast to float32 numpy but not transformed.
        * Remaining features are grouped into chunks of *chunk_size*, stacked
          into a single (T, chunk_size × S) matrix, transformed in one
          ``_numba_rolling_zscore_parallel`` dispatch, then unstacked.
        * Each chunk is freed before the next is processed, bounding peak
          memory to  chunk_size × T × S × 4 bytes  above the final output.

        Returns dict[str, np.ndarray] (float32, same shapes as input).
        """
        import gc

        keys_skip = []
        keys_xform = []
        for k in feats:
            if k in skip_keys:
                keys_skip.append(k)
            else:
                keys_xform.append(k)

        # --- Cast skipped features to numpy float32 ---
        for k in keys_skip:
            v = feats[k]
            feats[k] = np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32, copy=False)

        if not keys_xform:
            return feats

        # --- Process transformable features in chunks ---
        n_total = len(keys_xform)
        for chunk_start in range(0, n_total, chunk_size):
            chunk_keys = keys_xform[chunk_start:chunk_start + chunk_size]

            # Collect chunk arrays — each is (T, S)
            chunk_arrays = []
            for k in chunk_keys:
                v = feats[k]
                arr = np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32, copy=False)
                chunk_arrays.append(arr)
                feats[k] = None  # free original reference

            # Stack into (T, S*chunk) so _numba_rolling_zscore_parallel
            # processes all columns in one parallel dispatch
            stacked = np.concatenate(chunk_arrays, axis=1)  # (T, S*chunk)
            del chunk_arrays
            gc.collect()

            # Transform in-place where possible
            stacked = self._apply_transform_numpy(stacked)

            # Unstack back into per-feature arrays
            S = stacked.shape[1] // len(chunk_keys)
            for ci, k in enumerate(chunk_keys):
                feats[k] = np.ascontiguousarray(stacked[:, ci * S:(ci + 1) * S])
            del stacked
            gc.collect()

            tprint(f"  CausalTransform batch: {min(chunk_start + chunk_size, n_total)}/{n_total}")

        tprint(f"CausalTransform complete: {n_total} transformed, {len(keys_skip)} skipped")
        return feats''',
    '''    def transform_batch(
        self,
        feats: dict,
        skip_keys: set,
        chunk_size: int = 50,
    ) -> dict:
        """
        Transform all features in batched stacked-matrix calls, grouped by family.
        """
        import gc

        keys_skip = []
        family_groups = {}
        for k in feats:
            if k in skip_keys:
                keys_skip.append(k)
            else:
                family = get_feature_family(k)
                if family not in family_groups:
                    family_groups[family] = []
                family_groups[family].append(k)

        # --- Cast skipped features to numpy float32 ---
        for k in keys_skip:
            v = feats[k]
            feats[k] = np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32, copy=False)

        n_total = sum(len(v) for v in family_groups.values())
        if n_total == 0:
            return feats

        # --- Process transformable features by family and chunk ---
        processed_count = 0
        for family, keys_xform in family_groups.items():
            tprint(f"  CausalTransform processing family: {family} ({len(keys_xform)} features)")
            for chunk_start in range(0, len(keys_xform), chunk_size):
                chunk_keys = keys_xform[chunk_start:chunk_start + chunk_size]

                # Collect chunk arrays
                chunk_arrays = []
                for k in chunk_keys:
                    v = feats[k]
                    arr = np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32, copy=False)
                    chunk_arrays.append(arr)
                    feats[k] = None  # free original reference

                stacked = np.concatenate(chunk_arrays, axis=1)
                del chunk_arrays
                gc.collect()

                # Transform in-place using family policy
                stacked = self._apply_transform_numpy(stacked, family=family)

                S = stacked.shape[1] // len(chunk_keys)
                for ci, k in enumerate(chunk_keys):
                    feats[k] = np.ascontiguousarray(stacked[:, ci * S:(ci + 1) * S])
                del stacked
                gc.collect()

                processed_count += len(chunk_keys)
                tprint(f"  CausalTransform batch: {processed_count}/{n_total}")

        tprint(f"CausalTransform complete: {n_total} transformed across {len(family_groups)} families, {len(keys_skip)} skipped")
        return feats'''
)

with open('extreme_price_movements/feature_transforms.py', 'w') as f:
    f.write(content)
