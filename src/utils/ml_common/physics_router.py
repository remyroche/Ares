import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
import stumpy
from src.utils.ml_common.wavelet_utils import get_wavelet_features, wavelet_energy_ratios
from src.utils.tprint import tprint_info, tprint_warning, tprint_error
from src.utils.risk_regime_numba import rolling_mean_numba, rolling_std_numba, calculate_returns_numba
import hashlib
import pickle
import os
from functools import lru_cache

class AdaptiveHunterRouter:
    """
    Phase 1: Physics Router (Air Traffic Controller)
    Soft regime attribution using GMM on physics-based features.
    Optimized with Numba JIT and caching for performance.
    """
    def __init__(self, n_regimes: int = 3, base_smoothing: float = 0.85, window_size: int = 1000, mp_window: int = 30, cache_dir: str = ".cache/physics_router"):
        tprint_info(f"[AdaptiveHunterRouter] Initializing optimized router with n_regimes={n_regimes}, window_size={window_size}")
        self.n_regimes = n_regimes
        self.base_smoothing = base_smoothing
        self.window_size = window_size
        self.mp_window = mp_window
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        self.gmm: Optional[GaussianMixture] = None
        self.scaler = RobustScaler()
        self.regime_map: Dict[int, str] = {}

        self.last_weights: Optional[np.ndarray] = None
        self.log_lik_ema: Optional[float] = None
        self.log_lik_std: Optional[float] = None
        
        # Cache for expensive computations
        self._mp_cache: Dict[str, np.ndarray] = {}
        self._wavelet_cache: Dict[str, np.ndarray] = {}

        self.transition_matrix = np.array([
            [0.90, 0.08, 0.02],  # From Quiet
            [0.10, 0.85, 0.05],  # From Trending
            [0.05, 0.15, 0.80]   # From Chaos
        ])

    def compute_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 4 core physics features from Volume Bars.
        df must have: 'close', 'volume', 'bar_duration'
        """
        tprint_info(f"[AdaptiveHunterRouter] compute_physics_features start rows={len(df)}")
        if len(df) < self.mp_window * 2:
            return pd.DataFrame()

        # 1. Vol-Time Intensity
        vol_intensity = df['volume'] / (df['bar_duration'] + 1e-9)

        # 2. Efficiency Ratio - Optimized with Numba
        w_eff = 20
        close_returns = calculate_returns_numba(df['close'].values)
        direction = np.abs(np.convolve(close_returns, np.ones(w_eff), 'valid'))
        volatility = np.abs(close_returns)
        volatility_rolling = rolling_std_numba(volatility, w_eff) * np.sqrt(w_eff)
        
        # Pad to match original length
        efficiency_full = np.zeros(len(df))
        if len(direction) > 0 and len(volatility_rolling) >= len(direction):
            efficiency = direction / (volatility_rolling[:len(direction)] + 1e-9)
            efficiency_full[w_eff-1:] = efficiency
        
        efficiency = efficiency_full

        # 3. Matrix Profile Distance - Optimized with caching
        mp_dist = self._compute_matrix_profile_cached(df['close'].values)

        # 4. Wavelet Entropy - Vectorized computation
        wavelet_entropy = self._compute_wavelet_entropy_vectorized(df['close'].values)
        
        feats = pd.DataFrame(index=df.index)
        feats['vol_intensity'] = vol_intensity
        feats['efficiency'] = efficiency
        feats['mp_dist'] = mp_dist
        feats['wavelet_entropy'] = wavelet_entropy

        return feats.ffill().bfill()

    def _get_cache_key(self, data: np.ndarray, prefix: str) -> str:
        """Generate cache key based on data hash and parameters."""
        data_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]
        return f"{prefix}_{data_hash}_{self.window_size}_{self.mp_window}"

    def _compute_matrix_profile_cached(self, close_prices: np.ndarray) -> np.ndarray:
        """
        Compute matrix profile with caching for performance optimization.
        Uses sampling and interpolation to reduce computational load.
        """
        cache_key = self._get_cache_key(close_prices, "mp")
        
        # Check memory cache first
        if cache_key in self._mp_cache:
            tprint_info("[Router] Using cached matrix profile")
            return self._mp_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self._mp_cache[cache_key] = cached_data
                tprint_info("[Router] Loaded matrix profile from disk cache")
                return cached_data
            except Exception as e:
                tprint_warning(f"[Router] Failed to load cache: {e}")
        
        # Compute with optimized sampling
        tprint_info("[Router] Computing matrix profile with optimization...")
        try:
            step_mp = 5  # Sample every 5th point for ~4% computational load
            close_float = close_prices.astype(float)
            close_sampled = close_float[::step_mp]
            mp_window_sampled = max(3, self.mp_window // step_mp)
            
            if len(close_sampled) > mp_window_sampled * 2:
                mp = stumpy.stump(close_sampled, m=mp_window_sampled)
                mp_dist_sampled = mp[:, 0]
                
                # Vectorized expansion back to original length
                mp_dist = np.zeros(len(close_prices))
                indices = np.arange(len(close_sampled)) * step_mp
                valid_indices = indices[indices < len(close_prices)]
                
                # Fix: Properly align matrix profile with sampled indices
                # Matrix profile output is shorter than input by (window_size - 1)
                mp_start_offset = mp_window_sampled - 1
                mp_aligned_indices = valid_indices[mp_start_offset:]  # Skip first few indices
                
                # Ensure we do not exceed bounds
                max_assign_len = min(len(mp_aligned_indices), len(mp_dist_sampled))
                if max_assign_len > 0:
                    mp_dist[mp_aligned_indices[:max_assign_len]] = mp_dist_sampled[:max_assign_len]
                
                # Vectorized forward fill (using newer pandas syntax)
                mask = mp_dist == 0
                mp_dist = np.where(mask, np.nan, mp_dist)
                mp_dist = pd.Series(mp_dist).ffill().fillna(0).values
            else:
                mp_dist = np.zeros(len(close_prices))
                mp_dist = np.zeros(len(close_prices))
                
        except Exception as e:
            tprint_warning(f"[Router] Matrix profile computation failed: {e}. Using zeros.")
            mp_dist = np.zeros(len(close_prices))
        
        # Cache result
        self._mp_cache[cache_key] = mp_dist
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(mp_dist, f)
        except Exception as e:
            tprint_warning(f"[Router] Failed to save cache: {e}")
        
        return mp_dist

    def _compute_wavelet_entropy_vectorized(self, close_prices: np.ndarray) -> np.ndarray:
        """
        Vectorized wavelet entropy calculation to eliminate loops.
        Uses sliding windows with vectorized operations.
        """
        cache_key = self._get_cache_key(close_prices, "wavelet")
        
        # Check cache
        if cache_key in self._wavelet_cache:
            tprint_info("[Router] Using cached wavelet entropy")
            return self._wavelet_cache[cache_key]
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self._wavelet_cache[cache_key] = cached_data
                tprint_info("[Router] Loaded wavelet entropy from disk cache")
                return cached_data
            except Exception as e:
                tprint_warning(f"[Router] Failed to load wavelet cache: {e}")
        
        tprint_info("[Router] Computing wavelet entropy with vectorization...")
        
        # Vectorized sliding window approach
        n = len(close_prices)
        ratios = np.full(n, 0.5)
        
        if n >= self.window_size:
            # Create rolling windows efficiently
            windows = np.lib.stride_tricks.sliding_window_view(close_prices, self.window_size)
            
            # Sample windows for computation (every 5th window)
            step = min(5, len(windows) // 100)  # Ensure we have enough samples
            sampled_indices = np.arange(0, len(windows), step)
            sampled_ratios = np.zeros(len(sampled_indices))
            
            # Compute wavelet ratios for sampled windows
            for i, window_idx in enumerate(sampled_indices):
                window_data = windows[window_idx]
                try:
                    ratio = wavelet_energy_ratios(window_data, level=3)
                    sampled_ratios[i] = ratio
                except Exception:
                    sampled_ratios[i] = 0.5
            
            # Map sampled ratios back to original timeline
            sampled_positions = sampled_indices * step + self.window_size
            valid_positions = sampled_positions[sampled_positions < n]
            
            ratios[valid_positions] = sampled_ratios[:len(valid_positions)]
            
            # Vectorized forward fill
            ratios = pd.Series(ratios).fillna(method='ffill').fillna(0.5).values
        
        # Cache result
        self._wavelet_cache[cache_key] = ratios
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(ratios, f)
        except Exception as e:
            tprint_warning(f"[Router] Failed to save wavelet cache: {e}")
        
        return ratios

    def fit(self, X: np.ndarray):
        """
        Fit GMM on physics features to define regimes.
        X columns: [Vol_Intensity, Efficiency, MP_Dist, Wavelet_Entropy]
        """
        tprint_info(f"[AdaptiveHunterRouter] fit start samples={len(X)}")
        
        if X is None or len(X) < self.n_regimes * 2:
            tprint_warning(f"   [Router] Insufficient samples for fitting ({len(X) if X is not None else 0}). Skipping fit.")
            self.is_fit = False
            return self

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # GMM
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            reg_covar=1e-5,
            random_state=42
        ).fit(X_scaled)

        # Rank-Based Semantic Mapping
        means = self.gmm.means_
        vol_ranks = np.argsort(np.argsort(means[:, 0]))
        eff_ranks = np.argsort(np.argsort(means[:, 1]))
        mp_ranks = np.argsort(np.argsort(means[:, 2]))

        for i in range(self.n_regimes):
            if vol_ranks[i] <= 1 and mp_ranks[i] <= 1:
                self.regime_map[i] = "Quiet"
            elif eff_ranks[i] == (self.n_regimes - 1):
                self.regime_map[i] = "Trending"
            else:
                self.regime_map[i] = "Chaos"

        # Fallback
        used_labels = set(self.regime_map.values())
        if "Quiet" not in used_labels: self.regime_map[np.argmin(means[:, 0])] = "Quiet"
        if "Trending" not in used_labels: self.regime_map[np.argmax(means[:, 1])] = "Trending"

        tprint_info(f"   [Router] Regime Map: {self.regime_map}")

        scores = self.gmm.score_samples(X_scaled)
        self.log_lik_ema = np.mean(scores)
        self.log_lik_std = np.std(scores)

        self.is_fit = True

        return self

    def predict(self, x_current: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        tprint_info("[AdaptiveHunterRouter] predict called")
        if self.gmm is None or not getattr(self, 'is_fit', False):
            # Return uniform weights as fallback
            weights_final = np.ones(self.n_regimes) / self.n_regimes
            return weights_final, 0.0, 0.0, 0.0

        x_scaled = self.scaler.transform(x_current.reshape(1, -1))
        log_prob = self.gmm.score_samples(x_scaled)[0]
        weights_raw = self.gmm.predict_proba(x_scaled)[0]

        raw_entropy = -np.sum(weights_raw * np.log(weights_raw + 1e-9))
        max_ent = np.log(self.n_regimes)
        min_alpha = 0.2
        dynamic_alpha = min_alpha + (self.base_smoothing - min_alpha) * (1 - raw_entropy / max_ent)

        z_familiar = (log_prob - self.log_lik_ema) / (self.log_lik_std + 1e-9)
        chaos_boost = 0.4 * (1 / (1 + np.exp(z_familiar + 2.0)))

        chaos_idx = [k for k, v in self.regime_map.items() if v == "Chaos"]
        if not chaos_idx: chaos_idx = [self.n_regimes - 1]
        chaos_idx = chaos_idx[0]

        chaos_onehot = np.zeros(self.n_regimes)
        chaos_onehot[chaos_idx] = 1.0

        weights_blended = (1 - chaos_boost) * weights_raw + (chaos_boost * chaos_onehot)

        if self.last_weights is not None:
             predicted_weights = np.dot(self.last_weights, self.transition_matrix)
             raw_updated = predicted_weights * weights_blended
             weights_final = raw_updated / (np.sum(raw_updated) + 1e-9)
             weights_final = (dynamic_alpha * self.last_weights) + ((1 - dynamic_alpha) * weights_final)
        else:
            weights_final = weights_blended

        self.last_weights = weights_final

        if z_familiar > -3:
            self.log_lik_ema = 0.999 * self.log_lik_ema + 0.001 * log_prob

        router_confidence = (1 - raw_entropy / max_ent) * (1 / (1 + np.exp(-z_familiar)))

        return weights_final, raw_entropy, z_familiar, router_confidence
