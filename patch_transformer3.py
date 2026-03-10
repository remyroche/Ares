import re

with open('extreme_price_movements/feature_transforms.py', 'r') as f:
    content = f.read()

content = content.replace(
    '''    def transform(self, df: pd.DataFrame | np.ndarray, name: str = "unknown") -> pd.DataFrame | np.ndarray:
        """
        Applies Log + Causal Z-Score + Clip (Parametric Winsorization Proxy).
        O(N) complexity vs O(N*W) for rolling quantiles. ~300x Speedup.
        """
        if isinstance(df, np.ndarray):
            if df.size == 0:
                return df.copy()
            return self._apply_transform_numpy(df)

        if df.empty:
            return df.copy()

        df = df.sort_index()

        if not self.enable_cache:
            result = self._apply_transform_matrix(df)
            if self.debug:
                check_inf_nan(result, name)
            return result

        cached = self._load_cache(name)

        if cached is None:
            result = self._apply_transform_matrix(df)
            self._write_cache(name, result)
            if self.debug:
                check_inf_nan(result, name)
            return result

        cached_df = cached
        result = self._reuse_cache(df, cached_df)
        self._write_cache(name, result)

        if self.debug:
            check_inf_nan(result, name)

        return result''',
    '''    def transform(self, df: pd.DataFrame | np.ndarray, name: str = "unknown") -> pd.DataFrame | np.ndarray:
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

        return result'''
)

with open('extreme_price_movements/feature_transforms.py', 'w') as f:
    f.write(content)
