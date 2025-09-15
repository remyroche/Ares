"""
Top 20 Most Important Entropy Features

Entropy measures the information content and randomness in financial time series.
High entropy indicates more randomness, low entropy indicates more predictable patterns.

These are the most important entropy features for financial analysis:
1. Shannon Entropy - Classic information theory entropy
2. Rényi Entropy - Generalized entropy with different α values
3. Tsallis Entropy - Non-extensive entropy for complex systems
4. Sample Entropy - Measures complexity and regularity
5. Approximate Entropy - Similar to sample entropy but more robust
6. Permutation Entropy - Based on ordinal patterns
7. Wavelet Entropy - Time-frequency domain entropy
"""

import numpy as np
import pandas as pd
from typing import List
from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory

# 1. Price Entropy (Shannon Entropy) - Most fundamental
class PriceEntropyShannonGenerator(FeatureGenerator):
    """Generator for Shannon entropy of price movements."""
    
    def __init__(self, window: int = 20, bins: int = 10):
        config = FeatureConfig(
            name=f"price_entropy_shannon_{window}_{bins}",
            category=FeatureCategory.ENTROPY,
            description=f"Shannon entropy of price movements over {window} periods with {bins} bins",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'bins': bins}
        )
        super().__init__(config)
        self.window = window
        self.bins = bins
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def shannon_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                hist, _ = np.histogram(series.dropna(), bins=self.bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(shannon_entropy, raw=False)

# 2. Volume Entropy (Shannon Entropy)
class VolumeEntropyShannonGenerator(FeatureGenerator):
    """Generator for Shannon entropy of volume changes."""
    
    def __init__(self, window: int = 20, bins: int = 10):
        config = FeatureConfig(
            name=f"volume_entropy_shannon_{window}_{bins}",
            category=FeatureCategory.ENTROPY,
            description=f"Shannon entropy of volume changes over {window} periods with {bins} bins",
            required_columns=["volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'bins': bins}
        )
        super().__init__(config)
        self.window = window
        self.bins = bins
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        volume = data['volume']
        volume_returns = volume.pct_change()
        
        def shannon_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                hist, _ = np.histogram(series.dropna(), bins=self.bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        return volume_returns.rolling(window=self.window).apply(shannon_entropy, raw=False)

# 3. Return Entropy (Shannon Entropy)
class ReturnEntropyShannonGenerator(FeatureGenerator):
    """Generator for Shannon entropy of returns."""
    
    def __init__(self, window: int = 20, bins: int = 10):
        config = FeatureConfig(
            name=f"return_entropy_shannon_{window}_{bins}",
            category=FeatureCategory.ENTROPY,
            description=f"Shannon entropy of returns over {window} periods with {bins} bins",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'bins': bins}
        )
        super().__init__(config)
        self.window = window
        self.bins = bins
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def shannon_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                hist, _ = np.histogram(series.dropna(), bins=self.bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(shannon_entropy, raw=False)

# 4. Price Entropy (Rényi Entropy - α=2) - Emphasizes common events
class PriceEntropyRenyiGenerator(FeatureGenerator):
    """Generator for Rényi entropy (α=2) of price movements."""
    
    def __init__(self, window: int = 20, alpha: float = 2.0, bins: int = 10):
        config = FeatureConfig(
            name=f"price_entropy_renyi_{window}_{alpha}_{bins}",
            category=FeatureCategory.ENTROPY,
            description=f"Rényi entropy (α={alpha}) of price movements over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'alpha': alpha, 'bins': bins}
        )
        super().__init__(config)
        self.window = window
        self.alpha = alpha
        self.bins = bins
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def renyi_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                hist, _ = np.histogram(series.dropna(), bins=self.bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                if self.alpha == 1:
                    entropy = -np.sum(probs * np.log2(probs))
                else:
                    entropy = (1 / (1 - self.alpha)) * np.log2(np.sum(probs ** self.alpha))
                return entropy
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(renyi_entropy, raw=False)

# 5. Volume Entropy (Rényi Entropy - α=0.5) - Emphasizes rare events
class VolumeEntropyRenyiGenerator(FeatureGenerator):
    """Generator for Rényi entropy (α=0.5) of volume changes."""
    
    def __init__(self, window: int = 20, alpha: float = 0.5, bins: int = 10):
        config = FeatureConfig(
            name=f"volume_entropy_renyi_{window}_{alpha}_{bins}",
            category=FeatureCategory.ENTROPY,
            description=f"Rényi entropy (α={alpha}) of volume changes over {window} periods",
            required_columns=["volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'alpha': alpha, 'bins': bins}
        )
        super().__init__(config)
        self.window = window
        self.alpha = alpha
        self.bins = bins
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        volume = data['volume']
        volume_returns = volume.pct_change()
        
        def renyi_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                hist, _ = np.histogram(series.dropna(), bins=self.bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                if self.alpha == 1:
                    entropy = -np.sum(probs * np.log2(probs))
                else:
                    entropy = (1 / (1 - self.alpha)) * np.log2(np.sum(probs ** self.alpha))
                return entropy
            except:
                return 0.0
        
        return volume_returns.rolling(window=self.window).apply(renyi_entropy, raw=False)

# 6. Price Entropy (Tsallis Entropy) - Non-extensive entropy
class PriceEntropyTsallisGenerator(FeatureGenerator):
    """Generator for Tsallis entropy of price movements."""
    
    def __init__(self, window: int = 20, q: float = 2.0, bins: int = 10):
        config = FeatureConfig(
            name=f"price_entropy_tsallis_{window}_{q}_{bins}",
            category=FeatureCategory.ENTROPY,
            description=f"Tsallis entropy (q={q}) of price movements over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'q': q, 'bins': bins}
        )
        super().__init__(config)
        self.window = window
        self.q = q
        self.bins = bins
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def tsallis_entropy(series):
            if len(series) < 2:
                return 0.0
            try:
                hist, _ = np.histogram(series.dropna(), bins=self.bins)
                probs = hist / np.sum(hist)
                probs = probs[probs > 0]
                if self.q == 1:
                    entropy = -np.sum(probs * np.log2(probs))
                else:
                    entropy = (1 / (self.q - 1)) * (1 - np.sum(probs ** self.q))
                return entropy
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(tsallis_entropy, raw=False)

# 7. Price Entropy (Sample Entropy) - Measures complexity
class PriceEntropySampleGenerator(FeatureGenerator):
    """Generator for Sample entropy of price movements."""
    
    def __init__(self, window: int = 20, m: int = 2, r: float = 0.2):
        config = FeatureConfig(
            name=f"price_entropy_sample_{window}_{m}_{r}",
            category=FeatureCategory.ENTROPY,
            description=f"Sample entropy of price movements over {window} periods (m={m}, r={r})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'm': m, 'r': r}
        )
        super().__init__(config)
        self.window = window
        self.m = m
        self.r = r
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def sample_entropy(series):
            if len(series) < self.m + 1:
                return 0.0
            try:
                series = series.dropna().values
                N = len(series)
                r = self.r * np.std(series)
                
                def _maxdist(xi, xj, m):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _aproxm(N, m, r):
                    C = np.zeros(N - m + 1)
                    for i in range(N - m + 1):
                        template_i = series[i:i + m]
                        for j in range(N - m + 1):
                            template_j = series[j:j + m]
                            if _maxdist(template_i, template_j, m) <= r:
                                C[i] += 1.0
                    return C
                
                Cm = _aproxm(N, self.m, r)
                Cm1 = _aproxm(N, self.m + 1, r)
                
                phi = np.mean(np.log(Cm / (N - self.m + 1.0)))
                phi1 = np.mean(np.log(Cm1 / (N - self.m)))
                
                return phi - phi1
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(sample_entropy, raw=False)

# 8. Price Entropy (Approximate Entropy) - Similar to sample entropy
class PriceEntropyApproximateGenerator(FeatureGenerator):
    """Generator for Approximate entropy of price movements."""
    
    def __init__(self, window: int = 20, m: int = 2, r: float = 0.2):
        config = FeatureConfig(
            name=f"price_entropy_approximate_{window}_{m}_{r}",
            category=FeatureCategory.ENTROPY,
            description=f"Approximate entropy of price movements over {window} periods (m={m}, r={r})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'm': m, 'r': r}
        )
        super().__init__(config)
        self.window = window
        self.m = m
        self.r = r
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def approximate_entropy(series):
            if len(series) < self.m + 1:
                return 0.0
            try:
                series = series.dropna().values
                N = len(series)
                r = self.r * np.std(series)
                
                def _maxdist(xi, xj, m):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _aproxm(N, m, r):
                    C = np.zeros(N - m + 1)
                    for i in range(N - m + 1):
                        template_i = series[i:i + m]
                        for j in range(N - m + 1):
                            template_j = series[j:j + m]
                            if _maxdist(template_i, template_j, m) <= r:
                                C[i] += 1.0
                    return C
                
                Cm = _aproxm(N, self.m, r)
                Cm1 = _aproxm(N, self.m + 1, r)
                
                phi = np.mean(np.log(Cm / (N - self.m + 1.0)))
                phi1 = np.mean(np.log(Cm1 / (N - self.m)))
                
                return phi - phi1
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(approximate_entropy, raw=False)

# 9. Price Entropy (Permutation Entropy) - Based on ordinal patterns
class PriceEntropyPermutationGenerator(FeatureGenerator):
    """Generator for Permutation entropy of price movements."""
    
    def __init__(self, window: int = 20, m: int = 3):
        config = FeatureConfig(
            name=f"price_entropy_permutation_{window}_{m}",
            category=FeatureCategory.ENTROPY,
            description=f"Permutation entropy of price movements over {window} periods (m={m})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'm': m}
        )
        super().__init__(config)
        self.window = window
        self.m = m
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def permutation_entropy(series):
            if len(series) < self.m:
                return 0.0
            try:
                series = series.dropna().values
                N = len(series)
                
                # Generate all possible permutations
                from itertools import permutations
                perms = list(permutations(range(self.m)))
                perm_counts = {perm: 0 for perm in perms}
                
                # Count occurrences of each permutation pattern
                for i in range(N - self.m + 1):
                    subseries = series[i:i + self.m]
                    # Get the permutation pattern
                    perm_pattern = tuple(np.argsort(subseries))
                    if perm_pattern in perm_counts:
                        perm_counts[perm_pattern] += 1
                
                # Calculate probabilities and entropy
                total = sum(perm_counts.values())
                if total == 0:
                    return 0.0
                
                entropy = 0.0
                for count in perm_counts.values():
                    if count > 0:
                        p = count / total
                        entropy -= p * np.log2(p)
                
                return entropy
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(permutation_entropy, raw=False)

# 10. Price Entropy (Wavelet Entropy) - Time-frequency domain
class PriceEntropyWaveletGenerator(FeatureGenerator):
    """Generator for Wavelet entropy of price movements."""
    
    def __init__(self, window: int = 20, wavelet: str = 'db4'):
        config = FeatureConfig(
            name=f"price_entropy_wavelet_{window}_{wavelet}",
            category=FeatureCategory.ENTROPY,
            description=f"Wavelet entropy of price movements over {window} periods (wavelet={wavelet})",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'wavelet': wavelet}
        )
        super().__init__(config)
        self.window = window
        self.wavelet = wavelet
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def wavelet_entropy(series):
            if len(series) < 8:  # Minimum length for wavelet transform
                return 0.0
            try:
                # Simple wavelet-like decomposition using differences
                series = series.dropna().values
                N = len(series)
                
                # Approximate wavelet decomposition using Haar wavelet
                detail_coeffs = []
                for i in range(0, N-1, 2):
                    detail_coeffs.append(series[i+1] - series[i])
                
                if len(detail_coeffs) == 0:
                    return 0.0
                
                # Calculate energy distribution
                energies = np.array(detail_coeffs) ** 2
                total_energy = np.sum(energies)
                
                if total_energy == 0:
                    return 0.0
                
                # Normalize to probabilities
                probs = energies / total_energy
                probs = probs[probs > 0]
                
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs))
                return entropy
            except:
                return 0.0
        
        return returns.rolling(window=self.window).apply(wavelet_entropy, raw=False)

# 11-20. Additional entropy features with different parameters and windows
class PriceEntropyShannonShortGenerator(PriceEntropyShannonGenerator):
    def __init__(self):
        super().__init__(window=10, bins=8)

class PriceEntropyShannonLongGenerator(PriceEntropyShannonGenerator):
    def __init__(self):
        super().__init__(window=50, bins=15)

class VolumeEntropyShannonShortGenerator(VolumeEntropyShannonGenerator):
    def __init__(self):
        super().__init__(window=10, bins=8)

class VolumeEntropyShannonLongGenerator(VolumeEntropyShannonGenerator):
    def __init__(self):
        super().__init__(window=50, bins=15)

class ReturnEntropyShannonShortGenerator(ReturnEntropyShannonGenerator):
    def __init__(self):
        super().__init__(window=10, bins=8)

class ReturnEntropyShannonLongGenerator(ReturnEntropyShannonGenerator):
    def __init__(self):
        super().__init__(window=50, bins=15)

class PriceEntropyRenyiHighGenerator(PriceEntropyRenyiGenerator):
    def __init__(self):
        super().__init__(window=20, alpha=3.0, bins=12)

class PriceEntropyRenyiLowGenerator(PriceEntropyRenyiGenerator):
    def __init__(self):
        super().__init__(window=20, alpha=0.3, bins=12)

class VolumeEntropyRenyiHighGenerator(VolumeEntropyRenyiGenerator):
    def __init__(self):
        super().__init__(window=20, alpha=3.0, bins=12)

class VolumeEntropyRenyiLowGenerator(VolumeEntropyRenyiGenerator):
    def __init__(self):
        super().__init__(window=20, alpha=0.3, bins=12)

def create_top20_entropy_generators() -> List[FeatureGenerator]:
    """Create the top 20 most important entropy feature generators."""
    return [
        # Core entropy measures
        PriceEntropyShannonGenerator(20, 10),
        VolumeEntropyShannonGenerator(20, 10),
        ReturnEntropyShannonGenerator(20, 10),
        PriceEntropyRenyiGenerator(20, 2.0, 10),
        VolumeEntropyRenyiGenerator(20, 0.5, 10),
        PriceEntropyTsallisGenerator(20, 2.0, 10),
        PriceEntropySampleGenerator(20),
        PriceEntropyApproximateGenerator(20),
        PriceEntropyPermutationGenerator(20),
        PriceEntropyWaveletGenerator(20),
        
        # Variations with different parameters
        PriceEntropyShannonShortGenerator(),
        PriceEntropyShannonLongGenerator(),
        VolumeEntropyShannonShortGenerator(),
        VolumeEntropyShannonLongGenerator(),
        ReturnEntropyShannonShortGenerator(),
        ReturnEntropyShannonLongGenerator(),
        PriceEntropyRenyiHighGenerator(),
        PriceEntropyRenyiLowGenerator(),
        VolumeEntropyRenyiHighGenerator(),
        VolumeEntropyRenyiLowGenerator(),
    ]