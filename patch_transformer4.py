import re

with open('extreme_price_movements/feature_transforms.py', 'r') as f:
    content = f.read()

content = content.replace(
    '''    def _reuse_cache(self, df: pd.DataFrame, cached_df: pd.DataFrame) -> pd.DataFrame:
        try:
            cached_df = cached_df.sort_index()
        except Exception:
            return self._apply_transform_matrix(df)

        cached_len = len(cached_df)
        df_len = len(df)

        if cached_len == 0:
            return self._apply_transform_matrix(df)

        if cached_df.index[-1] > df.index[-1]:
            # Dataset shrank; safest to recompute.
            return self._apply_transform_matrix(df)

        if not df.index[:cached_len].equals(cached_df.index):
            # Index mismatch, fallback to full recompute
            return self._apply_transform_matrix(df)

        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)

        common_cols = [col for col in cached_df.columns if col in df.columns]
        if common_cols:
            result.loc[cached_df.index, common_cols] = cached_df[common_cols].to_numpy()

        new_cols = [col for col in df.columns if col not in cached_df.columns]
        if new_cols:
            new_transformed = self._apply_transform_matrix(df[new_cols])
            result.loc[:, new_cols] = new_transformed

        if df_len > cached_len:
            tail_start = max(0, cached_len - self.roll_window)
            tail_df = df.iloc[tail_start:]
            tail_transformed = self._apply_transform_matrix(tail_df)
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
            return self._apply_transform_matrix(df)

        return result''',
    '''    def _reuse_cache(self, df: pd.DataFrame, cached_df: pd.DataFrame, family: str = None) -> pd.DataFrame:
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

        return result'''
)

with open('extreme_price_movements/feature_transforms.py', 'w') as f:
    f.write(content)
