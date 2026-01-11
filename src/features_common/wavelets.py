
import numpy as np
import pandas as pd
import pywt
from typing import Optional, Union, List
from src.utils.tprint import tprint_info, tprint_warning

class RealTimeWaveletTransformer:
    """
    Strictly causal wavelet denoising transformer.
    
    Ensures that the denoised value at time t only depends on information
    available at or before time t (t, t-1, t-2, ...).
    
    Achieved by applying the wavelet transform on a rolling window basis
    and only retaining the last value (or causal reconstruction), 
    preventing any future leakage inherent in block-based transforms.
    """
    
    def __init__(self, wavelet: str = 'db4', level: int = 1, window_size: int = 64):
        self.wavelet = wavelet
        self.level = level
        self.window_size = window_size
        self._max_len = pywt.dwt_max_level(window_size, pywt.Wavelet(wavelet).dec_len)
        if level > self._max_len:
            tprint_warning(f"⚠️ Level {level} too high for window {window_size}. Clamping to {self._max_len}")
            self.level = self._max_len

    def transform(self, series: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Apply real-time wavelet denoising.
        
        Args:
            series: Input time series
            
        Returns:
            Denoised series (causal)
        """
        if isinstance(series, pd.Series):
            values = series.values
            index = series.index
        else:
            values = series
            index = None
            
        n = len(values)
        denoised = np.full(n, np.nan)
        
        # We need at least window_size data points to start
        if n < self.window_size:
            tprint_warning(f"⚠️ Series length {n} < window size {self.window_size}")
            return series if isinstance(series, pd.Series) else pd.Series(values)

        # Vectorized Rolling Window approach is hard with pywt which expects 1D array
        # We start loop from window_size
        
        # Optimization: For very large series, this loop is slow. 
        # But for 'real-time' simulation it is necessary.
        
        # Pre-allocate window buffer
        window = np.zeros(self.window_size)
        
        for i in range(self.window_size, n):
            # Extract window ending at i (exclusive of i, so indices i-window_size to i)
            # wait, if we want value at t=i, we need data up to i (inclusive)
            # So window is values[i-window_size+1 : i+1]
            
            window = values[i - self.window_size + 1 : i + 1]
            
            # Decompose
            try:
                # Use 'smooth' padding to minimize edge effects at the right boundary
                # though strictly the right boundary is T, so any padding extends into "future"
                # but "future" here is T+1 which doesn't exist.
                # 'periodization' is bad for trends. 'symmetric' is better.
                coeffs = pywt.wavedec(window, self.wavelet, mode='symmetric', level=self.level)
                
                # Thresholding
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(self.window_size))
                
                # Apply soft thresholding
                coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                
                # Reconstruct
                rec = pywt.waverec(coeffs, self.wavelet, mode='symmetric')
                
                # We only take the LAST point of the reconstruction
                # Because only the last point corresponds to time t
                # and uses the full context of the window up to t
                denoised[i] = rec[-1]
                
            except Exception:
                denoised[i] = values[i]

        # Fill initial NaN with original values (no history)
        denoised[:self.window_size] = values[:self.window_size]
        
        if index is not None:
            return pd.Series(denoised, index=index)
        return pd.Series(denoised)
