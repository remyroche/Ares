"""
Debug fixes for VOLATILITY_SPECIALIST and LIQUIDITY_SPECIALIST zero events issue.
"""

def add_debug_logging_to_volatility_specialist():
    """
    Add debug logging and adaptive thresholds to VolatilitySpecialistEvents.
    Key changes:
    1. Add debug logging to identify data issues
    2. Implement adaptive threshold based on actual volatility distribution
    3. Add fallback for near-constant volatility
    """
    patch_code = '''
    def _get_volatility_causal_events(self, df: pd.DataFrame, window=20, quantile=0.95):
        """Detect volatility expansion events using Parkinson range-based vol."""
        if 'close' not in df.columns:
            tprint_warning("VolatilitySpecialist: Missing 'close' column")
            return pd.DatetimeIndex([])
        
        # Debug logging
        tprint_info(f"VolatilitySpecialist: Input shape={df.shape}, columns={list(df.columns)}")
        tprint_info(f"VolatilitySpecialist: Price range={df['close'].min():.4f}-{df['close'].max():.4f}")
        
        # Use Parkinson volatility if high/low available, else fallback to close-based
        if 'high' in df.columns and 'low' in df.columns:
            log_hl = np.log(df['high'] / (df['low'] + 1e-9))
            parkinson_vol = log_hl / (2 * np.sqrt(np.log(2)))
            tprint_info("VolatilitySpecialist: Using Parkinson volatility (high/low)")
        else:
            ret = df['close'].pct_change()
            parkinson_vol = ret.abs()
            tprint_info("VolatilitySpecialist: Using close-based volatility (fallback)")
        
        # Check for constant prices (zero volatility)
        vol_std = parkinson_vol.std()
        if vol_std < 1e-8:
            tprint_warning(f"VolatilitySpecialist: Nearly zero volatility (std={vol_std:.2e})")
            # Generate events based on absolute price changes instead
            ret = df['close'].pct_change().abs()
            events = df.index[ret > ret.quantile(0.98)]
            tprint_info(f"VolatilitySpecialist: Fallback - {len(events)} events from price changes")
            return events
        
        tprint_info(f"VolatilitySpecialist: Vol stats mean={parkinson_vol.mean():.6f}, std={vol_std:.6f}")
        
        # Rest of original logic...
        vol_baseline = parkinson_vol.ewm(span=window * 5).mean()
        vol_ratio = parkinson_vol / (vol_baseline + 1e-9)
        
        lookback = min(100, len(df) // 3)  # Adaptive lookback
        shifted_mean = vol_ratio.shift(1).rolling(lookback).mean()
        shifted_std = vol_ratio.shift(1).rolling(lookback).std()
        z = (vol_ratio - shifted_mean) / (shifted_std + 1e-9)
        
        # Adaptive threshold: use actual data quantile if z-threshold too strict
        try:
            z_threshold = stats.norm.ppf(quantile)
            # If threshold would produce < 1% events, relax it
            potential_events = (z > z_threshold).sum()
            if potential_events < len(df) * 0.01:
                z_threshold = z.quantile(0.98)  # Use empirical 98th percentile
                tprint_info(f"VolatilitySpecialist: Relaxed threshold to {z_threshold:.3f}")
        except:
            z_threshold = z.quantile(0.98)
        
        events = df.index[z > z_threshold]
        tprint_info(f"VolatilitySpecialist: Generated {len(events)} events")
        
        return events
    '''
    return patch_code

def add_debug_logging_to_liquidity_specialist():
    """
    Add debug logging and adaptive thresholds to LiquiditySpecialistEvents.
    Key changes:
    1. Add debug logging for data validation
    2. Implement adaptive threshold based on actual liquidity distribution
    3. Add volume validation checks
    """
    patch_code = '''
    def _get_liquidity_causal_events(self, df: pd.DataFrame, window=20, threshold=2.5):
        """Detect liquidity IMPROVEMENT events (favorable entry conditions)."""
        if 'volume' not in df.columns or 'close' not in df.columns:
            tprint_warning("LiquiditySpecialist: Missing required columns")
            return pd.DatetimeIndex([])
        
        # Debug logging
        tprint_info(f"LiquiditySpecialist: Input shape={df.shape}, columns={list(df.columns)}")
        tprint_info(f"LiquiditySpecialist: Volume range={df['volume'].min()}-{df['volume'].max()}")
        
        # Check volume data quality
        if df['volume'].std() < 1e-6:
            tprint_warning("LiquiditySpecialist: Nearly constant volume")
            return pd.DatetimeIndex([])
        
        ret = df['close'].pct_change()
        
        # Amihud illiquidity (price impact per volume)
        amihud = ret.abs() / (df['volume'] + 1e-9)
        liquidity = 1.0 / (amihud + 1e-9)
        
        # Check liquidity data quality
        if liquidity.std() < 1e-6:
            tprint_warning("LiquiditySpecialist: Nearly constant liquidity")
            return pd.DatetimeIndex([])
        
        # Adaptive threshold based on actual distribution
        liq_mean = liquidity.rolling(window * 5).mean()
        liq_std = liquidity.rolling(window * 5).std()
        liq_z = (liquidity - liq_mean) / (liq_std + 1e-9)
        
        # Kyle's lambda
        vol_signed = df['volume'] * np.sign(ret)
        cov_window = min(window * 2, len(df) // 4)
        cov = ret.rolling(cov_window).cov(vol_signed)
        var = vol_signed.rolling(cov_window).var()
        kyle_lambda = cov / (var + 1e-9)
        
        lambda_mean = kyle_lambda.rolling(window * 5).mean()
        lambda_std = kyle_lambda.rolling(window * 5).std()
        lambda_z = (kyle_lambda - lambda_mean) / (lambda_std + 1e-9)
        
        # Adaptive threshold: if too strict, use empirical quantiles
        liq_events = (liq_z > threshold).sum()
        lambda_events = (lambda_z < -threshold).sum()
        total_events = liq_events + lambda_events
        
        if total_events < len(df) * 0.01:  # Less than 1% events
            # Relax thresholds
            liq_threshold = liq_z.quantile(0.98)
            lambda_threshold = lambda_z.quantile(0.02)  # Lower values = better liquidity
            tprint_info(f"LiquiditySpecialist: Relaxed thresholds - liq: {liq_threshold:.3f}, lambda: {lambda_threshold:.3f}")
        else:
            liq_threshold = threshold
            lambda_threshold = -threshold
        
        mask = (liq_z > liq_threshold) | (lambda_z < lambda_threshold)
        events = df.index[mask]
        tprint_info(f"LiquiditySpecialist: Generated {len(events)} events")
        
        return events
    '''
    return patch_code

print("Debug fixes prepared for VOLATILITY_SPECIALIST and LIQUIDITY_SPECIALIST")
print("Key improvements:")
print("1. Added comprehensive debug logging")
print("2. Implemented adaptive thresholds based on data distribution")
print("3. Added fallback mechanisms for edge cases")
print("4. Added data quality validation")
