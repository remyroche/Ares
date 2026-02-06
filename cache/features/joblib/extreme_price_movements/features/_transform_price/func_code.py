# first line: 33
@_cache.cache
def _transform_price(df, _label=""):
    tprint(f"Transforming Prices ({_label}): Log -> EWMA(5) -> Adaptive FracDiff [{df.shape[1]} cols]")
    # Safe Log: Clip input to be at least 1e-9 to avoid log(0) or log(neg)
    df_log = np.log(np.maximum(df, 1e-9))
    df_den = ff.apply_to_frame(df_log, ff._numba_ewma_nan_safe, 2.0/6.0, False)
    
    # Apply adaptive FFD per column
    df_fd = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float32)
    total_cols = len(df_den.columns)
    for i, col in enumerate(df_den.columns):
        if (i+1) % 5 == 0 or (i+1) == total_cols:
             tprint(f"Adaptive FFD ({_label}): {i+1}/{total_cols} - {col}")
        
        series = df_den[col].dropna()
        if len(series) < 100:
            # Fallback to fixed d=0.4 for short series
            d_opt = 0.4
        else:
            # Find minimal d for stationarity
            d_opt, _, _ = find_min_ffd(series, d_range=(0.0, 1.0), step=0.1)
        
        # Apply FFD
        df_fd[col] = frac_diff_ffd(df_den[col], d_opt, thres=1e-5)
    
    tprint(f"Adaptive FFD ({_label}): d range [{df_fd.min().min():.3f}, {df_fd.max().max():.3f}]")
    return df_fd
