# first line: 220
@_cache.cache
def _transform_volume(df):
    tprint("Transforming Volume: Log -> EWMA(5)")
    df_log = np.log(df + 1.0)
    df_den = ff.numba_ewma(df_log, 2.0/6.0, False)
    return df_den
