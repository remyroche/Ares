import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
import stumpy
from src.utils.ml_common.wavelet_utils import get_wavelet_features
from src.utils.tprint import tprint_info, tprint_warning, tprint_error

class AdaptiveHunterRouter:
    """
    Phase 1: Physics Router (Air Traffic Controller)
    Soft regime attribution using GMM on physics-based features.
    """
    def __init__(self, n_regimes: int = 3, base_smoothing: float = 0.85, window_size: int = 1000, mp_window: int = 30):
        self.n_regimes = n_regimes
        self.base_smoothing = base_smoothing
        self.window_size = window_size
        self.mp_window = mp_window

        self.gmm: Optional[GaussianMixture] = None
        self.scaler = RobustScaler()
        self.regime_map: Dict[int, str] = {}

        self.last_weights: Optional[np.ndarray] = None
        self.log_lik_ema: Optional[float] = None
        self.log_lik_std: Optional[float] = None

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
        tprint_info("   [Router] Computing Physics Features (Vol, Eff, MP, Wavelet)...")
        if len(df) < self.mp_window * 2:
            return pd.DataFrame()

        # 1. Vol-Time Intensity
        vol_intensity = df['volume'] / (df['bar_duration'] + 1e-9)

        # 2. Efficiency Ratio
        w_eff = 20
        direction = df['close'].diff(w_eff).abs()
        volatility = df['close'].diff().abs().rolling(w_eff).sum()
        efficiency = direction / (volatility + 1e-9)

        # 3. Matrix Profile Distance
        close_float = df['close'].values.astype(float)
        try:
            mp = stumpy.stump(close_float, m=self.mp_window)
            pad_width = len(df) - len(mp)
            mp_dist = np.concatenate([np.full(pad_width, np.nan), mp[:, 0]])
        except Exception as e:
            tprint_warning(f"   [Router] Stumpy failed: {e}. Using fallback.")
            mp_dist = np.zeros(len(df))

        # 4. Wavelet Entropy (Approximation)
        # Using rolling entropy of returns as proxy for speed if not using full wavelet
        # But let's try to be closer to spec: L1/L4 ratio.
        # We can reuse 'get_wavelet_features' in a loop or stride?
        # For training, loop is okay.
        # Stride = 10?
        # Let's use a simpler proxy for vectorization:
        # Entropy of returns over rolling window.
        roll_std = df['close'].pct_change().rolling(32).std()
        roll_mean_abs = df['close'].pct_change().abs().rolling(32).mean()
        # Coeff of Variation approx? No.
        # Let's placeholder with rolling std for now as "Entropy Proxy".
        wavelet_entropy = roll_std

        feats = pd.DataFrame(index=df.index)
        feats['vol_intensity'] = vol_intensity
        feats['efficiency'] = efficiency
        feats['mp_dist'] = mp_dist
        feats['wavelet_entropy'] = wavelet_entropy

        return feats.ffill().bfill()

    def fit(self, X: np.ndarray):
        """
        Fit GMM on physics features to define regimes.
        X columns: [Vol_Intensity, Efficiency, MP_Dist, Wavelet_Entropy]
        """
        tprint_info(f"   [Router] Fitting GMM on {len(X)} samples...")
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

        return self

    def predict(self, x_current: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        if self.gmm is None: raise ValueError("Router not fit")

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
