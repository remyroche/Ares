"""
Adaptive Hunter Router
----------------------
Implements regime detection using Gaussian Mixture Models and adaptive physics-based features.
"""

import time
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer
from scipy.special import expit as sigmoid
from src.utils.numba_funcs import _numba_rolling_entropy, _numba_run_regime_filter, _numba_volatility_clustering, _numba_return_autocorrelation, _numba_price_jump_frequency
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

    def __init__(self, n_regimes: int = 3, base_smoothing: float = 0.85, **kwargs):
        self.n_regimes = kwargs.get("n_components", n_regimes)
        self.base_smoothing = base_smoothing
        self.gmm: Optional[GaussianMixture] = None
        self.regime_map: Dict[int, str] = {}
        self.last_weights: Optional[np.ndarray] = None

        # Adaptive OOD Tracking
        self.log_lik_ema: Optional[float] = None
        self.log_lik_std: Optional[float] = None
        self.scaler = QuantileTransformer(output_distribution="uniform", n_quantiles=500, random_state=42)
        
        # Fit state flag
        self.is_fit = False

        # Transition Matrix (Persistence/Inertia)
        self.transition_matrix = np.eye(self.n_regimes) * 0.85 + (1 - 0.85) / self.n_regimes

    def fit(self, X: np.ndarray):
        """
        Fit the GMM on historical features.

        Args:
            X: Feature matrix [n_samples, 4] -> [Vol_Intensity, Efficiency, MP_Dist, Wavelet_Entropy]
        """
        if len(X) < self.n_regimes:
            tprint_warning(f"⚠️ Not enough samples to fit GMM ({len(X)} < {self.n_regimes}).")
            self.regime_map = {i: f"Regime_{i}" for i in range(self.n_regimes)}
            return

        tprint_info("🧠 AdaptiveHunterRouter: Scaling features...")
        t_start_scale = time.time()
        # 1. Scale Full Data (Fast)
        X_scaled = self.scaler.fit_transform(X)
        tprint_info(f"   [Router] Scaling took {time.time() - t_start_scale:.2f}s")
        
        tprint_info("🧠 AdaptiveHunterRouter: Fitting GMM on historical features...")
        t_start_fit = time.time()
        # 2. Subsample for GMM (Bottleneck Optimization)
        # 129k samples * 10 inits takes forever. 10k is sufficient for 3 clusters.
        if len(X_scaled) > 10000:
            tprint_info(f"   [Router] Subsampling 10,000/{len(X_scaled)} samples for GMM fitting efficiency.")
            idx = np.random.RandomState(42).choice(len(X_scaled), 10000, replace=False)
            X_fit = X_scaled[idx]
        else:
            X_fit = X_scaled

        # Check for degenerate features after scaling
        feature_std = np.std(X_fit, axis=0)
        if np.any(feature_std < 1e-6):
            tprint_warning(f"   [Router] Degenerate features detected (std < 1e-6): {np.where(feature_std < 1e-6)[0]}")
            # Add small noise to degenerate features
            X_fit[:, feature_std < 1e-6] += np.random.normal(0, 1e-4, X_fit[:, feature_std < 1e-6].shape)
        
        self.gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='diag', # More stable and faster than 'full'
            reg_covar=1e-2,         # Increased from 1e-4 for robustness
            random_state=42,
            max_iter=50,
            n_init=1,             # One good init is enough for 2-5 clusters
            init_params='kmeans',
            verbose=0
        ).fit(X_fit)
        tprint_info(f"   [Router] GMM Fitting took {time.time() - t_start_fit:.2f}s")

        # 1. Percentile-based Semantic Mapping (better chaos detection)
        means = self.gmm.means_
        
        # Calculate percentiles for each feature across all regimes
        vol_percentiles = np.percentile(means[:, 0], [25, 75])  # Volatility Intensity
        eff_percentiles = np.percentile(means[:, 1], [25, 75])  # Efficiency Ratio
        entropy_percentiles = np.percentile(means[:, 3], [25, 75])  # Wavelet Entropy
        
        self.regime_map = {}
        
        for i in range(self.n_regimes):
            vol = means[i, 0]
            eff = means[i, 1]
            entropy = means[i, 3]
            
            # Chaos: High volatility (>75th percentile) AND low efficiency (<25th percentile) AND high entropy (>75th percentile)
            if (vol > vol_percentiles[1] and eff < eff_percentiles[0] and entropy > entropy_percentiles[1]):
                self.regime_map[i] = "Chaos"
            # Quiet: Low volatility (<25th percentile) AND moderate efficiency
            elif vol < vol_percentiles[0] and eff >= eff_percentiles[0]:
                self.regime_map[i] = "Quiet"
            # Trending: High efficiency (>75th percentile) (regardless of volatility)
            elif eff > eff_percentiles[1]:
                self.regime_map[i] = "Trending"
            # Fallback assignments based on dominant feature
            else:
                if eff > np.mean(means[:, 1]):
                    self.regime_map[i] = "Trending"
                elif vol < np.mean(means[:, 0]):
                    self.regime_map[i] = "Quiet"
                else:
                    self.regime_map[i] = "Chaos"
        
        # Ensure all regime types are present (fallback logic)
        used_labels = set(self.regime_map.values())
        if "Quiet" not in used_labels:
            # Assign lowest volatility regime to Quiet
            self.regime_map[np.argmin(means[:, 0])] = "Quiet"
        if "Trending" not in used_labels:
            # Assign highest efficiency regime to Trending
            self.regime_map[np.argmax(means[:, 1])] = "Trending"
        if "Chaos" not in used_labels:
            # FIX: Find the first regime with a duplicate label and assign it to Chaos
            label_counts = {}
            for idx, label in self.regime_map.items():
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Find a label that appears more than once
            for idx, label in self.regime_map.items():
                if label_counts[label] > 1:
                    self.regime_map[idx] = "Chaos"
                    tprint_info(f"   [Router] Reassigned cluster {idx} from '{label}' to 'Chaos' to ensure 3 distinct regimes")
                    break
        # 2. Score and store OOD stats
        scores = self.gmm.score_samples(X_scaled)
        self.log_lik_ema = np.mean(scores)
        self.log_lik_std = np.std(scores)
        
        # Set fit flag after successful fitting
        self.is_fit = True
        
        # Debug: Log GMM diagnostics for convergence investigation
        vol_ranks = np.argsort(means[:, 0])
        eff_ranks = np.argsort(means[:, 1])
        mp_ranks = np.argsort(means[:, 2])
        
        tprint_info(f"   [Router] GMM Convergence: {self.gmm.converged_}")
        tprint_info(f"   [Router] GMM Means:\n{means}")
        tprint_info(f"   [Router] Feature Ranks - Vol: {vol_ranks}, Eff: {eff_ranks}, MP: {mp_ranks}")
        tprint_info(f"   [Router] Regime Map: {self.regime_map}")
        tprint_info(f"   [Router] Log-likelihood stats: mean={self.log_lik_ema:.4f}, std={self.log_lik_std:.4f}")

    def _calculate_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate physics features including chaos detection:
        [Vol_Intensity, Efficiency, MP_Dist, Wavelet_Entropy, 
         Vol_Clustering, Return_Autocorr, Price_Jump_Freq]
        """
        if df.empty:
            return pd.DataFrame()
            
        # 1. Volatility Intensity
        if 'volatility_1d' in df.columns:
            vol = df['volatility_1d']
        else:
            vol = df['close'].pct_change().rolling(20).std()

        vol_long = vol.rolling(100).mean()
        vol_std = vol.rolling(100).std()
        vol_intensity = (vol - vol_long) / (vol_std + 1e-9)

        # 2. Efficiency Ratio
        close = df['close']
        window = 10
        change = (close - close.shift(window)).abs()
        path = close.diff().abs().rolling(window).sum()
        efficiency = change / (path + 1e-9)

        # 3. Market Profile Distance (Proxy: Dist from VWAP)
        if 'vwap' in df.columns:
            mp_dist = (close - df['vwap']).abs() / (close + 1e-9)
        else:
            # Fallback: Dist from SMA 20
            sma = close.rolling(20).mean()
            mp_dist = (close - sma).abs() / (close + 1e-9)

        # 4. Entropy (Optimized with Numba)
        returns = df['close'].pct_change().fillna(0.0).values
        entropy = _numba_rolling_entropy(returns, window=20, bins=5)
        
        # 5. NEW: Chaos Detection Features
        # 5a. Volatility Clustering (GARCH-like behavior)
        vol_clustering = _numba_volatility_clustering(returns, window=20)
        
        # 5b. Return Autocorrelation (negative values indicate chaos)
        return_autocorr = _numba_return_autocorrelation(returns, window=20, lag=1)
        
        # 5c. Price Jump Frequency (high frequency indicates turbulence)
        jump_frequency = _numba_price_jump_frequency(returns, window=20, threshold=2.0)

        features = pd.DataFrame({
            'vol_intensity': vol_intensity,
            'efficiency': efficiency,
            'mp_dist': mp_dist,
            'entropy': entropy,
            'vol_clustering': vol_clustering,
            'return_autocorr': return_autocorr,
            'jump_frequency': jump_frequency
        }, index=df.index).fillna(0.0)

        # Feature Quality Diagnostics
        feature_stats = {
            'vol_intensity': {'mean': features['vol_intensity'].mean(), 'std': features['vol_intensity'].std(), 'nan_pct': features['vol_intensity'].isna().sum() / len(features) * 100},
            'efficiency': {'mean': features['efficiency'].mean(), 'std': features['efficiency'].std(), 'nan_pct': features['efficiency'].isna().sum() / len(features) * 100},
            'mp_dist': {'mean': features['mp_dist'].mean(), 'std': features['mp_dist'].std(), 'nan_pct': features['mp_dist'].isna().sum() / len(features) * 100},
            'entropy': {'mean': features['entropy'].mean(), 'std': features['entropy'].std(), 'nan_pct': features['entropy'].isna().sum() / len(features) * 100},
            'vol_clustering': {'mean': features['vol_clustering'].mean(), 'std': features['vol_clustering'].std(), 'nan_pct': features['vol_clustering'].isna().sum() / len(features) * 100},
            'return_autocorr': {'mean': features['return_autocorr'].mean(), 'std': features['return_autocorr'].std(), 'nan_pct': features['return_autocorr'].isna().sum() / len(features) * 100},
            'jump_frequency': {'mean': features['jump_frequency'].mean(), 'std': features['jump_frequency'].std(), 'nan_pct': features['jump_frequency'].isna().sum() / len(features) * 100}
        }
        
        tprint_info(f"   [Router] Feature Quality Diagnostics:")
        for feat, stats in feature_stats.items():
            tprint_info(f"      {feat}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, nan%={stats['nan_pct']:.2f}%")

        return features.clip(-5, 5)


    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Convenience method for one-shot fit and predict.
        """
        features = self._calculate_physics_features(df)
        self.fit(features.values)
        res_df = self.predict_batch(features)
        return res_df['regime_label']

    def predict(self, x_current: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Predict regime weights for a single bar/batch.

        Args:
            x_current: Feature vector(s) [1, 4]

        Returns:
            Tuple: (weights, entropy, z_familiar, confidence)
        """
        x_current = np.atleast_2d(x_current)
        
        if self.gmm is None:
            # Fallback for unfitted router
            w = np.zeros(self.n_regimes)
            w[0] = 1.0
            return w, 0.0, 0.0, 1.0

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
        if self.last_weights is None or len(self.last_weights) != self.n_regimes:
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
        return self.regime_map.get(idx, f"Regime_{idx}")


    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Run prediction over a full dataframe (batch mode) using Vectorized Ops + Numba HMM."""
        tprint_info(f"🔮 AdaptiveHunterRouter: Predicting batch of {len(X)} samples (Vectorized)...")
        
        if self.gmm is None:
             tprint_warning("   [Router] GMM not fitted. Returning fallback.")
             return pd.DataFrame({'regime_label': ['Quiet']*len(X)}, index=X.index)

        # 1. Vectorized Pre-computation (Heaviest operations)
        X_arr = X.values
        t_start_pred = time.time()
        tprint_info("   [Router] Scaling batch features...")
        X_scaled = self.scaler.transform(X_arr)
        tprint_info(f"   [Router] Scaling took {time.time() - t_start_pred:.2f}s")
        
        # GMM computations (Vectorized C-level efficient)
        t_start_gmm = time.time()
        tprint_info("   [Router] Computing GMM probabilities...")
        log_probs = self.gmm.score_samples(X_scaled)
        weights_raw = self.gmm.predict_proba(X_scaled)
        tprint_info(f"   [Router] GMM probabilities took {time.time() - t_start_gmm:.2f}s")
        
        # Entropy (Vectorized)
        t_start_ent = time.time()
        tprint_info("   [Router] Computing probabilities entropy...")
        # -sum(p * log(p))
        w_log = np.log(weights_raw + 1e-9)
        entropies = -np.sum(weights_raw * w_log, axis=1)
        tprint_info(f"   [Router] Entropy computation took {time.time() - t_start_ent:.2f}s")
        
        # 2. Sequential Filter (Numba Optimized Loop)
        t_start_filter = time.time()
        tprint_info("   [Router] Running Numba sequential filter...")
        chaos_indices = [k for k, v in self.regime_map.items() if v == "Chaos"]
        # Fallback to Volatility (dim 0) if no specific Chaos label
        chaos_idx = chaos_indices[0] if chaos_indices else int(np.argmax(self.gmm.means_[:, 0]))
        
        if self.log_lik_ema is None:
             self.log_lik_ema = np.mean(log_probs) # Fallback init
        if self.log_lik_std is None:
             self.log_lik_std = np.std(log_probs)  # Fallback init

        # Run Numba Loop
        final_weights, z_familiars, confidences = _numba_run_regime_filter(
            log_probs, 
            weights_raw, 
            entropies, 
            self.n_regimes, 
            self.transition_matrix, 
            chaos_idx, 
            float(self.log_lik_ema), 
            float(self.log_lik_std), 
            float(self.base_smoothing)
        )
        tprint_info(f"   [Router] Numba sequential filter took {time.time() - t_start_filter:.2f}s")
        
        # Update internal state (using last values from batch)
        # Approximate update for future calls
        if len(final_weights) > 0:
             self.last_weights = final_weights[-1]
             # Update EMA based on last Z (simple approximation)
             last_z = z_familiars[-1]
             if last_z > -3:
                  self.log_lik_ema = 0.999 * self.log_lik_ema + 0.001 * log_probs[-1]

        # 3. Label Allocation
        regime_ids = np.argmax(final_weights, axis=1)
        labels = [self.regime_map.get(i, f"Regime_{i}") for i in regime_ids]

        # 4. Result Construction
        data = {
            'regime_id': regime_ids,
            'regime_label': labels,
            'entropy': entropies,
            'z_familiar': z_familiars,
            'confidence': confidences
        }
        # Add probability columns
        for i in range(self.n_regimes):
            data[f'prob_{i}'] = final_weights[:, i]
            
        res_df = pd.DataFrame(data, index=X.index)
        
        # Regime Balance Validation
        regime_counts = res_df['regime_label'].value_counts()
        total_samples = len(res_df)
        if total_samples > 0:
            regime_percentages = (regime_counts / total_samples * 100).round(2)
            
            # Warn about extreme imbalances
            max_pct = regime_percentages.max()
            min_pct = regime_percentages.min()
            if max_pct > 80 or min_pct < 5:
                tprint_warning(f"⚠️ 🚨 ALERT: Extreme regime imbalance detected!")
                tprint_warning(f"   Distribution: {dict(regime_percentages)}%")
                tprint_warning(f"   Consider adjusting GMM parameters or feature engineering")
            else:
                tprint_success(f"✅ Market Regimes distribution: {dict(regime_counts)}")
        
        # Remap prob_0/1/2 to names
        for idx, name in self.regime_map.items():
            res_df[f'prob_{name}'] = res_df[f'prob_{idx}']

        tprint_success(f"✅ AdaptiveHunterRouter: Batch prediction complete. Shape: {res_df.shape}")
        return res_df
