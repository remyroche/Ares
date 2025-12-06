class UnifiedOHLCVStandardizer:
    """Fallback no-op standardizer for OHLCV data."""

    def standardize(self, df, *args, **kwargs):
        return df
