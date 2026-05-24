import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import extreme_price_movements.fast_funcs as ff
from scipy.stats import norm
from .utils import tprint, check_inf_nan
from .feature_family_registry import get_feature_family, FeatureFamily

class CausalFeatureTransformer:
    _CACHE_VERSION = 1

    def __init__(
        self,
        winsor_qt=0.02,
        roll_window=24 * 30,
        debug=False,
        cache_dir: str | os.PathLike | None = None,
        enable_cache: bool = True,
    ):
        tprint(f"Entering function: __init__ in feature_transforms.py")
        self.winsor_qt = winsor_qt
        self.roll_window = roll_window
        self.debug = debug
        self.enable_cache = enable_cache
        # Precompute sigma threshold for clipping (assuming Normality after Log)
        # For two-sided clipping at winsor_qt (e.g. 0.02 -> 1% top/bottom? Or 2% total?)
        # Original code used dual quantiles: winsor_qt and (1-winsor_qt).
        # So we want to clip at prob=winsor_qt and prob=1-winsor_qt.
        # This matches norm.ppf(1 - winsor_qt) for the upper bound.
        self.sigma_k = float(norm.ppf(1.0 - winsor_qt))
        default_cache_dir = Path("./cache/feature_transforms")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        tprint(
            f"CausalFeatureTransformer: Optimized Parametric Mode (sigma={self.sigma_k:.3f})"
        )

    def transform(self, df: pd.DataFrame | np.ndarray, name: str = "unknown") -> pd.DataFrame | np.ndarray:
        """
        Applies family-aware transforms (e.g. Log + Causal Z-Score + Clip).
        """
        family = get_feature_family(name)

        if isinstance(df, np.ndarray):
            if df.size == 0:
                return df.copy()
            return self._apply_transform_numpy(df, family=family)

        if df.empty:
            return df.copy()

        df = df.sort_index()

        if not self.enable_cache:
            result = self._apply_transform_matrix(df, family=family)
            if self.debug:
                check_inf_nan(result, name)
            return result

        cached = self._load_cache(name)

        if cached is None:
            result = self._apply_transform_matrix(df, family=family)
            self._write_cache(name, result)
            if self.debug:
                check_inf_nan(result, name)
            return result

        cached_df = cached
        result = self._reuse_cache(df, cached_df, family=family)
        self._write_cache(name, result)

        if self.debug:
            check_inf_nan(result, name)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_transform_numpy(self, mat: np.ndarray, family: str = None) -> np.ndarray:
        """Core transform on a contiguous float32 numpy array (T, C). Returns new array."""
        if family == FeatureFamily.CATEGORICAL_OR_BUCKETED:
            return mat.copy()

        mat = np.ascontiguousarray(mat, dtype=np.float32)
        input_bad = ~np.isfinite(mat)

        do_arcsinh = (family == FeatureFamily.RISK_NORMALIZED_CONTINUOUS)
        do_zscore = (family == FeatureFamily.RISK_NORMALIZED_CONTINUOUS)
        do_clip = (family in [FeatureFamily.RISK_NORMALIZED_CONTINUOUS, FeatureFamily.ALREADY_STANDARDIZED, FeatureFamily.BOUNDED_GEOMETRY])

        if do_arcsinh:
            np.arcsinh(mat, out=mat)

        if do_zscore:
            mat = ff._numba_rolling_zscore_parallel(mat, self.roll_window)

        np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        if do_clip:
            np.clip(mat, -self.sigma_k, self.sigma_k, out=mat)
        if input_bad.any():
            mat[input_bad] = np.nan

        return mat

    def _apply_transform_matrix(self, df: pd.DataFrame, family: str = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        mat = self._apply_transform_numpy(df.to_numpy(dtype=np.float32, copy=False), family=family)
        return pd.DataFrame(mat, index=df.index, columns=df.columns)

    def transform_batch(
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
            for chunk_idx, chunk_start in enumerate(range(0, len(keys_xform), chunk_size)):
                chunk_keys = keys_xform[chunk_start:chunk_start + chunk_size]

                # Separate 1D (market-level) from 2D (per-symbol) features
                keys_2d = []
                keys_1d = []
                for k in chunk_keys:
                    v = feats[k]
                    arr = np.asarray(v, dtype=np.float32) if not isinstance(v, np.ndarray) else v.astype(np.float32, copy=False)
                    feats[k] = arr
                    if arr.ndim >= 2:
                        keys_2d.append(k)
                    else:
                        keys_1d.append(k)

                tprint(f"    Processing chunk of {len(chunk_keys)} features: {chunk_keys[:5]}...")
                import time
                t0 = time.time()

                # Transform 1D features individually (preserves shape)
                for k in keys_1d:
                    arr = feats[k]
                    feats[k] = None
                    feats[k] = self._apply_transform_numpy(
                        arr[:, np.newaxis], family=family
                    ).ravel()

                # Batch-transform 2D features via stacking
                if keys_2d:
                    chunk_arrays = []
                    chunk_widths = []
                    for k in keys_2d:
                        arr = feats[k]
                        chunk_arrays.append(arr)
                        chunk_widths.append(arr.shape[1])
                        feats[k] = None

                    stacked = np.concatenate(chunk_arrays, axis=1)
                    split_offsets = np.cumsum([0, *chunk_widths], dtype=np.int32)
                    del chunk_arrays
                    gc.collect()

                    stacked = self._apply_transform_numpy(stacked, family=family)

                    for ci, k in enumerate(keys_2d):
                        start = int(split_offsets[ci])
                        end = int(split_offsets[ci + 1])
                        feats[k] = np.ascontiguousarray(stacked[:, start:end])
                    del stacked

                if (chunk_idx + 1) % 4 == 0 or (chunk_start + chunk_size) >= len(keys_xform):
                    gc.collect()

                t1 = time.time()

                processed_count += len(chunk_keys)
                tprint(f"  CausalTransform batch: {processed_count}/{n_total} (family={family}, chunk_time={t1-t0:.2f}s)")

        tprint(f"CausalTransform complete: {n_total} transformed across {len(family_groups)} families, {len(keys_skip)} skipped")
        return feats

    def _reuse_cache(self, df: pd.DataFrame, cached_df: pd.DataFrame, family: str = None) -> pd.DataFrame:
        try:
            cached_df = cached_df.sort_index()
        except Exception:
            return self._apply_transform_matrix(df, family=family)

        cached_len = len(cached_df)
        df_len = len(df)

        if cached_len == 0:
            return self._apply_transform_matrix(df, family=family)

        if cached_df.index[-1] > df.index[-1]:
            # Dataset shrank; safest to recompute.
            return self._apply_transform_matrix(df, family=family)

        if not df.index[:cached_len].equals(cached_df.index):
            # Index mismatch, fallback to full recompute
            return self._apply_transform_matrix(df, family=family)

        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)

        common_cols = [col for col in cached_df.columns if col in df.columns]
        if common_cols:
            result.loc[cached_df.index, common_cols] = cached_df[common_cols].to_numpy()

        new_cols = [col for col in df.columns if col not in cached_df.columns]
        if new_cols:
            new_transformed = self._apply_transform_matrix(df[new_cols], family=family)
            result.loc[:, new_cols] = new_transformed

        if df_len > cached_len:
            tail_start = max(0, cached_len - self.roll_window)
            tail_df = df.iloc[tail_start:]
            tail_transformed = self._apply_transform_matrix(tail_df, family=family)
            result.iloc[tail_start:] = tail_transformed.to_numpy()
            tprint(
                f"CausalFeatureTransformer: reused {cached_len} rows, computed {df_len - cached_len} new rows"
            )
        else:
            tprint(
                f"CausalFeatureTransformer: cache hit for '{df.columns[0] if len(df.columns)==1 else 'batch'}', no new rows"
            )

        if result.isna().any().any():
            # Safety: fall back to full computation if any gaps remain
            return self._apply_transform_matrix(df, family=family)

        return result

    def _sanitize_name(self, name: str) -> str:
        safe = re.sub(r"[^0-9a-zA-Z_.-]", "_", str(name))
        if not safe:
            return "feature"
        return safe

    def _cache_paths(self, name: str) -> tuple[Path, Path, Path]:
        safe_name = self._sanitize_name(name)
        feature_dir = self.cache_dir / safe_name
        data_path = feature_dir / "transformed.parquet"
        meta_path = feature_dir / "meta.json"
        return feature_dir, data_path, meta_path

    def _load_cache(self, name: str) -> pd.DataFrame | None:
        if not self.enable_cache:
            return None

        feature_dir, data_path, meta_path = self._cache_paths(name)
        if not data_path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path, "r") as fp:
                meta = json.load(fp)
        except Exception:
            return None

        if meta.get("version") != self._CACHE_VERSION:
            return None

        if meta.get("roll_window") != self.roll_window:
            return None

        if float(meta.get("winsor_qt", -1)) != float(self.winsor_qt):
            return None

        try:
            cached_df = pd.read_parquet(data_path)
            return cached_df
        except Exception:
            return None

    def _write_cache(self, name: str, df: pd.DataFrame) -> None:
        if not self.enable_cache or df.empty:
            return

        feature_dir, data_path, meta_path = self._cache_paths(name)
        feature_dir.mkdir(parents=True, exist_ok=True)

        tmp_data = data_path.with_suffix(".tmp.parquet")
        tmp_meta = meta_path.with_suffix(".tmp")

        df_to_store = df.astype(np.float32)
        df_to_store.to_parquet(tmp_data)
        os.replace(tmp_data, data_path)

        try:
            first_ts = df_to_store.index[0].isoformat() if len(df_to_store) else None
            last_ts = df_to_store.index[-1].isoformat() if len(df_to_store) else None
        except AttributeError:
            first_ts = str(df_to_store.index[0]) if len(df_to_store) else None
            last_ts = str(df_to_store.index[-1]) if len(df_to_store) else None

        meta = {
            "version": self._CACHE_VERSION,
            "roll_window": self.roll_window,
            "winsor_qt": float(self.winsor_qt),
            "sigma_k": float(self.sigma_k),
            "rows": int(len(df_to_store)),
            "cols": list(df_to_store.columns),
            "first_ts": first_ts,
            "last_ts": last_ts,
        }

        with open(tmp_meta, "w") as fp:
            json.dump(meta, fp)

        os.replace(tmp_meta, meta_path)

def log_winsor_zscore_rolling(series: pd.Series, window: int = 720, qt: float = 0.02) -> pd.Series:
    """Helper for single series causal transform (Parametric)"""
    # 1. To numpy
    arr = series.to_numpy(dtype=np.float32, copy=False)

    # 2. Arcsinh
    arr = np.arcsinh(arr)

    # 3. Z-Score
    arr = ff._numba_rolling_zscore_nan_safe_1d(arr, window)

    # 4. Clip
    sigma = float(norm.ppf(1.0 - qt))
    np.clip(arr, -sigma, sigma, out=arr)

    return pd.Series(arr, index=series.index, name=series.name)
