import re

with open('extreme_price_movements/feature_transforms.py', 'r') as f:
    content = f.read()

content = content.replace(
    '''    def _apply_transform_numpy(self, mat: np.ndarray) -> np.ndarray:
        """Core transform on a contiguous float32 numpy array (T, C). Returns new array."""
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        np.arcsinh(mat, out=mat)
        mat = ff._numba_rolling_zscore_parallel(mat, self.roll_window)
        np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.clip(mat, -self.sigma_k, self.sigma_k, out=mat)
        return mat

    def _apply_transform_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        mat = self._apply_transform_numpy(df.to_numpy(dtype=np.float32, copy=False))
        return pd.DataFrame(mat, index=df.index, columns=df.columns)''',
    '''    def _apply_transform_numpy(self, mat: np.ndarray, family: str = None) -> np.ndarray:
        """Core transform on a contiguous float32 numpy array (T, C). Returns new array."""
        if family == FeatureFamily.CATEGORICAL_OR_BUCKETED:
            return mat.copy()

        mat = np.ascontiguousarray(mat, dtype=np.float32)

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

        return mat

    def _apply_transform_matrix(self, df: pd.DataFrame, family: str = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        mat = self._apply_transform_numpy(df.to_numpy(dtype=np.float32, copy=False), family=family)
        return pd.DataFrame(mat, index=df.index, columns=df.columns)'''
)

with open('extreme_price_movements/feature_transforms.py', 'w') as f:
    f.write(content)
