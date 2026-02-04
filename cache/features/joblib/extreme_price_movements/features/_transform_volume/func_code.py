# first line: 55
@_cache.cache
def _transform_volume(df):
    tprint("Transforming Volume: Log -> EWMA(5)")
    df_log = np.log(df + 1.0)
    df_den = ff.apply_to_frame(df_log, ff._numba_ewma_nan_safe, 2.0/6.0, False)
    return df_den
