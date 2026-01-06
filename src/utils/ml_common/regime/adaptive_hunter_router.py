"""
Adaptive Hunter Router
----------------------
Implements regime detection using Gaussian Mixture Models and adaptive physics-based features.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from scipy.special import expit as sigmoid
from typing import Tuple, Dict, Any, Optional
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class AdaptiveHunterRouter:
    """
    Adaptive Hunter Router for market regime detection.

    Identifies market regimes (Quiet, Trending, Chaos) based on physics features:
    - Volatility Intensity
    - Efficiency Ratio
    - Market Profile Distance
    - Wavelet Entropy

    Uses GMM for clustering and an adaptive forward filter for state estimation.
    """

    def __init__(self, n_regimes: int = 3, base_smoothing: float = 0.85):
        self.n_regimes = n_regimes
        self.base_smoothing = base_smoothing
        self.gmm: Optional[GaussianMixture] = None
        self.regime_map: Dict[int, str] = {}
        self.last_weights: Optional[np.ndarray] = None

        # Adaptive OOD Tracking
        self.log_lik_ema: Optional[float] = None
        self.log_lik_std: Optional[float] = None
        self.scaler = RobustScaler()

        # Transition Matrix (Persistence/Inertia)
        self.transition_matrix = np.array([
            [0.90, 0.08, 0.02],  # From Quiet -> [Quiet, Trending, Chaos]
            [0.10, 0.85, 0.05],  # From Trending -> [Quiet, Trending, Chaos]
            [0.05, 0.15, 0.80]   # From Chaos -> [Quiet, Trending, Chaos]
        ])

    def fit(self, X: np.ndarray):
        """
        Fit the GMM on historical features.

        Args:
            X: Feature matrix [n_samples, 4] -> [Vol_Intensity, Efficiency, MP_Dist, Wavelet_Entropy]
        """
        tprint_info("🧠 AdaptiveHunterRouter: Fitting GMM on historical features...")
        X_scaled = self.scaler.fit_transform(X)
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            reg_covar=1e-5,
            random_state=42
        ).fit(X_scaled)

        # 1. Rank-Based Semantic Mapping (Removes Scale Bias)
        means = self.gmm.means_
        # Ranks: Lower index = Lower value
        # Features: 0=Vol, 1=Eff, 2=MP_Dist, 3=Entropy
        vol_ranks = np.argsort(np.argsort(means[:, 0]))
        eff_ranks = np.argsort(np.argsort(means[:, 1]))
        mp_ranks  = np.argsort(np.argsort(means[:, 2]))

        self.regime_map = {}
        for i in range(self.n_regimes):
            # Quiet: Low Vol + Low MP distance (Motif presence)
            if vol_ranks[i] == 0 and mp_ranks[i] == 0:
                self.regime_map[i] = "Quiet"
            # Trending: High Vol + High Efficiency
            elif vol_ranks[i] == (self.n_regimes-1) and eff_ranks[i] == (self.n_regimes-1):
                self.regime_map[i] = "Trending"
            else:
                self.regime_map[i] = "Chaos"

        # Fill in gaps if mapping logic missed any (ensure all indices covered)
        assigned_names = set(self.regime_map.values())
        if len(self.regime_map) < self.n_regimes:
            for i in range(self.n_regimes):
                if i not in self.regime_map:
                    if "Quiet" not in assigned_names: self.regime_map[i] = "Quiet"
                    elif "Trending" not in assigned_names: self.regime_map[i] = "Trending"
                    else: self.regime_map[i] = "Chaos"

        # 2. Initialize Adaptive OOD Baselines
        scores = self.gmm.score_samples(X_scaled)
        self.log_lik_ema = np.mean(scores)
        self.log_lik_std = np.std(scores)

    def predict(self, x_current: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Predict regime weights for a single bar/batch.

        Args:
            x_current: Feature vector(s) [1, 4]

        Returns:
            Tuple: (weights, entropy, z_familiar, confidence)
        """
        x_current = np.atleast_2d(x_current)
        x_scaled = self.scaler.transform(x_current)

        # Log likelihood of sample
        log_prob = self.gmm.score_samples(x_scaled)[0]

        # 3. Entropy-Aware Inertia
        weights_raw = self.gmm.predict_proba(x_scaled)[0]
        raw_entropy = -np.sum(weights_raw * np.log(weights_raw + 1e-9))

        # Dynamic smoothing: If confused (high entropy), reduce inertia
        min_alpha = 0.2
        max_entropy = np.log(self.n_regimes)
        dynamic_alpha = min_alpha + (self.base_smoothing - min_alpha) * (1 - raw_entropy / (max_entropy + 1e-9))

        # 4. Adaptive OOD (Relative Familiarity)
        z_familiar = (log_prob - self.log_lik_ema) / (self.log_lik_std + 1e-9)

        # Calibrated Chaos Boost: Sigmoid based on 2-sigma deviation
        # If z_familiar < -2 (unfamiliar), boost chaos
        chaos_boost = 0.4 * sigmoid(-(z_familiar + 2.0))

        # Find Chaos Index
        chaos_indices = [k for k, v in self.regime_map.items() if v == "Chaos"]
        chaos_idx = chaos_indices[0] if chaos_indices else np.argmax(self.gmm.means_[:, 0]) # Fallback: Highest Vol

        chaos_onehot = np.zeros(self.n_regimes)
        chaos_onehot[chaos_idx] = 1.0

        # Blend OOD boost
        weights_blended = (1 - chaos_boost) * weights_raw + (chaos_boost * chaos_onehot)

        # 5. Forward Filter Update
        if self.last_weights is None:
            self.last_weights = weights_blended
        else:
            # Use forward filter logic
            predicted_weights = np.dot(self.last_weights, self.transition_matrix)
            evidence = weights_blended
            updated = predicted_weights * evidence
            self.last_weights = updated / (np.sum(updated) + 1e-9)

        # Update rolling OOD stats (slowly) to handle non-stationarity
        if z_familiar > -3:
            self.log_lik_ema = 0.999 * self.log_lik_ema + 0.001 * log_prob

        router_confidence = (1 - raw_entropy / max_entropy) * sigmoid(z_familiar)

        return self.last_weights, raw_entropy, z_familiar, router_confidence

    def get_regime_label(self, weights: np.ndarray) -> str:
        """Get the string label for the dominant regime."""
        idx = np.argmax(weights)
        return self.regime_map.get(idx, "Unknown")

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run prediction over a full dataframe (batch mode)."""
        tprint_info(f"🔮 AdaptiveHunterRouter: Predicting batch of {len(X)} samples...")
        results = []
        # Reset state for batch prediction
        self.last_weights = None
        # We assume X is sorted by time
        X_arr = X.values

        for i in range(len(X_arr)):
            w, ent, z, conf = self.predict(X_arr[i])
            label = self.get_regime_label(w)
            results.append({
                'regime_id': np.argmax(w),
                'regime_label': label,
                'prob_quiet': w[0] if 0 in self.regime_map and self.regime_map[0]=="Quiet" else 0.0,
                # Note: this mapping is fragile if indices shuffle.
                # Better:
                'prob_0': w[0],
                'prob_1': w[1],
                'prob_2': w[2],
                'entropy': ent,
                'z_familiar': z,
                'confidence': conf
            })

        res_df = pd.DataFrame(results, index=X.index)
        tprint_success(f"✅ AdaptiveHunterRouter: Batch prediction complete. Shape: {res_df.shape}")

        # Remap prob_0/1/2 to names
        for idx, name in self.regime_map.items():
            res_df[f'prob_{name}'] = res_df[f'prob_{idx}']

        return res_df
